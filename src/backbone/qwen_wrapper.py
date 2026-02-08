import os
import torch
from PIL import Image
from typing import List, Union, Dict, Any
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class QwenUnifiedBackbone:
    def __init__(self, model_path: str = None, device_map: str = "auto"):
        self.model_path = model_path or os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.device_map = device_map

        print(f"Loading model from {self.model_path}...")
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                device_map=device_map,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        
        # 预先获取 Yes/No 的 Token ID
        # 为了稳健性，获取 "Yes" 和 "No" 的 id。
        # 注意：不同 tokenizer 可能对首字母大写敏感，且前面可能有空格。
        # 这里我们假设简单的 "Yes", "No"。
        # Qwen2 Tokenizer 通常不需要 add_special_tokens=False 来获取单词本身的 id，但为了保险起见。
        # 更好的方式是看 vocab，但在代码里我们尝试 encode。
        self.yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]
        
        print("Model loaded successfully.")
        
        # 冻结模型
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def get_vqa_scores(self, image: Union[Image.Image, str], candidates: List[str]) -> Dict[str, float]:
        """
        批量计算候选词的存在概率。
        
        Args:
            image: PIL Image 或 图片路径
            candidates: 候选词列表 (concept names)
            
        Returns:
            Dict {concept: score}
        """
        scores = {}
        if not candidates:
            return scores
        
        # 构造 Batch Inputs
        # 提示词模板: "Is there a {c} in this image? Answer Yes or No."
        
        messages_batch = []
        for c in candidates:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"Is there a {c} in this image? Answer Yes or No."},
                    ],
                }
            ]
            messages_batch.append(messages)
            
        # 准备输入
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_batch
        ]
        
        image_inputs, video_inputs = process_vision_info(messages_batch)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Forward Pass (No Generate)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 获取最后一个 Token 的 Logits
        # Qwen2VL 是 Causal LM，最后一个 input token 的 output logits 预测下一个 token。
        # batch 里的每个样本长度可能不同（因为 padding），我们需要取出每个样本实际有效长度的最后一个 logits。
        # 如果是左 padding，最后一个就是 -1。如果是右 padding (transformers默认)，需要找 attention_mask。
        
        # 简化处理：transformers 的 pad_token_id 通常在右边。
        # 我们可以通过 input_ids 找到最后一个非 pad token。
        
        next_token_logits = []
        for i in range(len(texts)):
            # 找到最后一个非 pad 的 index
            input_ids = inputs.input_ids[i]
            # attention_mask 为 1 的最后一个位置
            last_idx = (inputs.attention_mask[i] == 1).nonzero(as_tuple=True)[0][-1]
            logits = outputs.logits[i, last_idx, :]
            next_token_logits.append(logits)
            
        # 计算 Yes/No 的概率
        final_scores = {}
        for idx, concept in enumerate(candidates):
            logits = next_token_logits[idx]
            yes_score = logits[self.yes_token_id].item()
            no_score = logits[self.no_token_id].item()
            
            # Softmax on (Yes, No)
            # prob_yes = exp(yes) / (exp(yes) + exp(no))
            # 简化计算，直接用 exp
            import math
            exp_yes = math.exp(yes_score)
            exp_no = math.exp(no_score)
            prob_yes = exp_yes / (exp_yes + exp_no)
            
            final_scores[concept] = prob_yes
            
        return final_scores

    def generate_text_response(self, messages: List[Dict[str, Any]], max_new_tokens: int = 128) -> str:
        """
        [新增] 纯文本生成接口，用于 Concept Extraction 和 Query Parsing
        """
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def generate_mixed_response(self, text_contexts: List[str], image_paths: List[str], user_query: str) -> str:
        """
        [新增] 混合模态推理接口
        
        Args:
            text_contexts: 检索到的相关文本段落列表
            image_paths: 检索到的图片路径列表
            user_query: 用户原始问题
        """
        content_payload = []
        
        # 1. 注入文本知识
        if text_contexts:
            content_payload.append({"type": "text", "text": "Relevant Knowledge from Documents:\n"})
            for i, txt in enumerate(text_contexts):
                content_payload.append({"type": "text", "text": f"Doc {i+1}: {txt}\n"})
            content_payload.append({"type": "text", "text": "\n"})

        # 2. 注入视觉上下文 (Interleaved Images)
        if image_paths:
            content_payload.append({"type": "text", "text": "Relevant Visual Context:\n"})
            for i, img_path in enumerate(image_paths):
                # Qwen2-VL Processor 直接接受 path 字符串或 PIL Image
                # 为了防止文件不存在导致的错误，建议在调用此函数前做好过滤，或者在这里 try-catch
                try:
                    content_payload.append({"type": "image", "image": img_path})
                    content_payload.append({"type": "text", "text": f" (Image {i+1})\n"})
                except Exception as e:
                    print(f"Warning: enhancing prompt with image {img_path} failed: {e}")
            content_payload.append({"type": "text", "text": "\n"})
            
        # 3. 注入用户问题
        # 系统提示词强化
        sys_instruction = "Based on the provided visual and textual context, please answer the question."
        content_payload.append({"type": "text", "text": f"{sys_instruction}\nQuestion: {user_query}"})
        
        messages = [
            {
                "role": "user",
                "content": content_payload
            }
        ]
        
        # 复用通用的 generate_response
        return self.generate_response(messages)

    def extract_embeddings(self, image: Union[Image.Image, str]) -> torch.Tensor:
        """
        提取图像的视觉特征。
        返回的是 Visual Encoder 的输出，经过 Projector 之后的 Embedding 的均值。
        """
        # 构造一个 dummy message 来触发 processor 的图像处理
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        
        image_inputs, video_inputs = process_vision_info(messages)
        # 我们只需要图像处理后的 pixel_values
        inputs = self.processor(
            text=[""], # Dummy text, we rely on visual path
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            if hasattr(self.model, "visual"):
                # Qwen2VL visual forward
                # 根据 modeling_qwen2_vl.py，visual 返回 grid_thw 等信息
                # model.visual(pixel_values, grid_thw=image_grid_thw)
                pixel_values = inputs.pixel_values
                image_grid_thw = inputs.image_grid_thw
                
                # 获取 visual embeddings (before projection or after?)
                # Qwen2VL 的 visual 模块通常直接输出 hidden states
                visual_output = self.model.visual(pixel_values, grid_thw=image_grid_thw)
                
                # 处理返回值：可能是 tuple 或 tensor
                if isinstance(visual_output, tuple):
                    # 如果是 tuple，通常第一个元素是 embeddings
                    visual_embeds = visual_output[0]
                else:
                    # 如果是 tensor，直接使用
                    visual_embeds = visual_output
                
                # 处理维度：确保是 [total_pixels, hidden_size] 格式
                # 如果有多余的 batch 维度，先 squeeze
                if len(visual_embeds.shape) == 3:
                    # [batch, total_pixels, hidden_size] -> [total_pixels, hidden_size]
                    visual_embeds = visual_embeds.squeeze(0)
                elif len(visual_embeds.shape) == 2:
                    # [total_pixels, hidden_size] - 正确格式
                    pass
                else:
                    raise ValueError(f"Unexpected visual_embeds shape: {visual_embeds.shape}")
                
                # Qwen2VL 的 visual_embeds 通常是 [total_pixels, hidden_size]
                # 它不是 batch 形式的，而是把所有图片的 patch 拼在一起（基于 flash attn 的优化）
                # 如果我们只输入了一张图片，那么 visual_embeds 就是这张图的所有 patch
                # 我们可以直接做 mean pooling
                
                return visual_embeds.mean(dim=0).cpu() # Shape: [hidden_size]
            else:
                # Fallback 或者报错
                raise NotImplementedError("Cannot find visual module in model.")

    def generate_response(self, prompt_messages: List[Dict], images: List[Union[Image.Image, str]] = None) -> str:
        """
        通用 Chat 接口。
        prompt_messages 应该符合 Qwen 的 Chat 格式。
        """
        texts = [
            self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        ]
        
        image_inputs, video_inputs = process_vision_info(prompt_messages)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
