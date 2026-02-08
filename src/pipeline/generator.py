import torch
from typing import List, Dict, Tuple, Union, Any
from PIL import Image


class Generator:
    def __init__(self, backbone: Any):
        self.backbone = backbone

    def construct_prompt(
        self,
        global_image: Union[Image.Image, str],
        patches: List[Tuple[Image.Image, List[int]]],
        query: str,
    ) -> List[Dict]:
        """
        构建生成 Prompt。

        注意：按当前实验设定，主图推理阶段**永远不传入 patches**（Local Slices），
        以避免多图导致序列过长/显存溢出，并保持不同模型的输入一致性。

        Template:
        <|im_start|>system
        You are an expert in visual reasoning. I will provide a global view image. Analyze it carefully to answer the question.
        <|im_end|>
        <|im_start|>user
        [Global View]: <image_placeholder>
        Question: {user_query}
        <|im_end|>
        """

        # 构造 User Content 部分
        user_content = []

        # Global View
        user_content.append({"type": "text", "text": "[Global View]: "})
        user_content.append({"type": "image", "image": global_image})

        # Question
        user_content.append({"type": "text", "text": f"\nQuestion: {query}"})

        # 组装完整 Messages
        # System Message 需要被 Backbone 正确处理。
        # transformers 的 apply_chat_template 通常支持 system role。
        # 将 system content 转换为列表格式，与 user content 保持一致
        system_content = [
            {
                "type": "text",
                "text": "You are an expert in visual reasoning. I will provide a global view image. Analyze it carefully to answer the question.",
            }
        ]

        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": user_content},
        ]

        return messages

    def generate(
        self,
        global_image: Union[Image.Image, str],
        patches: List[Tuple[Image.Image, List[int]]],
        query: str,
        max_new_tokens: Union[int, None] = None,
        stop_strings: Union[List[str], str, None] = None,
    ) -> str:
        messages = self.construct_prompt(global_image, patches, query)

        # Generate Response（支持 max_new_tokens / stop_strings 限制输出）
        kwargs = {}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        if stop_strings is not None:
            kwargs["stop_strings"] = stop_strings
        response = self.backbone.generate_response(messages, **kwargs)
        return response

    def generate_answer(
        self,
        retrieval_results: Dict[str, List],
        query: str,
        image_root: str = None,
    ) -> str:
        """
        [新增] 针对 RAG 检索结果的生成接口

        Args:
            retrieval_results: {'images': [{'image_id':...}, ...], 'texts': [{'content':...}, ...]}
            query: User's question
            image_root: Directory where raw images are stored
        """
        import os

        if image_root is None:
            image_root = os.environ.get("IMAGE_ROOT", ".")

        # 1. 提取文本内容
        text_contexts = []
        for txt_item in retrieval_results.get("texts", []):
            content = txt_item.get("content", "")
            # 简单的过滤：忽略过短的文本
            if len(content.strip()) > 10:
                text_contexts.append(content)

        # 2. 提取图片路径
        image_paths = []
        for img_item in retrieval_results.get("images", []):
            img_id = img_item.get("image_id")
            # 假设文件名是 {id}.jpg，需要根据实际情况适配
            # 如果 id 已经是路径或包含后缀，则需调整
            # 为了兼容性，先尝试查找 .jpg, .png, .jpeg
            found = False
            for ext in [".jpg", ".png", ".jpeg"]:
                p = os.path.join(image_root, f"{img_id}{ext}")
                if os.path.exists(p):
                    image_paths.append(p)
                    found = True
                    break

            # 如果没找到，也可以尝试直接把 id 当路径 (如果 id 本身就是相对或绝对路径)
            if not found and os.path.exists(img_id):
                image_paths.append(img_id)

        # 3. 调用 Backbone 进行混合推理
        # 如果既没有图也没有文，则只回答问题 (退化为 LLM)
        if not text_contexts and not image_paths:
            print("Warning: No context retrieved for generation.")

        response = self.backbone.generate_mixed_response(
            text_contexts, image_paths, query
        )
        return response

    def generate_with_relationships(
        self,
        main_image: Union[Image.Image, str],
        patches: List[Tuple[Image.Image, List[int]]],
        query: str,
        retrieved_images: Dict[str, Image.Image],
        relationship_context: str = "",
    ) -> str:
        """
        [新增] 生成接口，支持多图 + 关系上下文 (Option B)
        按当前实验设定：主图阶段**不传入 patches**（Local Slices）。

        Args:
            main_image: 主图片
            patches: 兼容保留（当前不会使用）
            query: 用户问题
            retrieved_images: {'retrieved_img_id': PIL_Image, ...}
            relationship_context: 图像关系描述字符串
        """
        content_payload = []

        # 1. 首先展示主图片（不包含 patches）
        content_payload.append({"type": "text", "text": "## QUESTION IMAGE (Main Image to Answer):\n"})
        content_payload.append({"type": "text", "text": "[Global View]: "})
        content_payload.append({"type": "image", "image": main_image})

        # 2. 然后展示检索到的参考图片（明确标注为参考）
        if retrieved_images:
            content_payload.append({"type": "text", "text": "\n## REFERENCE IMAGES (Retrieved for Context - NOT the question image):\n"})
            for idx, (img_id, img) in enumerate(retrieved_images.items(), 1):
                content_payload.append({"type": "text", "text": f"Reference {idx} ({img_id}): "})
                content_payload.append({"type": "image", "image": img})
                content_payload.append({"type": "text", "text": "\n"})

        # 3. 关系上下文（如果有）
        if relationship_context:
            content_payload.append(
                {
                    "type": "text",
                    "text": f"\n## Image Relationships:\n{relationship_context}\n",
                }
            )

        # 4. 问题
        content_payload.append({"type": "text", "text": "\n## Question:\n"})
        content_payload.append({"type": "text", "text": query})

        # 根据是否有检索图片，使用不同的 system prompt
        if retrieved_images:
            system_content_text = (
                "You are an expert in visual reasoning. I will provide:\n"
                "1. A QUESTION IMAGE (global view) - This is the primary image you need to analyze.\n"
                "2. REFERENCE IMAGES from a knowledge base - These provide additional context and examples.\n\n"
                "Your task: Carefully examine the QUESTION IMAGE first, then use the REFERENCE IMAGES to find relevant patterns, "
                "features, or concepts that can help you answer. Compare and analyze both to determine the correct answer."
            )
        else:
            system_content_text = (
                "You are an expert in visual reasoning. I will provide a question image (global view). "
                "Analyze the image carefully to answer the question."
            )
        
        # 将 system content 转换为列表格式，与 user content 保持一致
        system_content = [{"type": "text", "text": system_content_text}]
        
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": content_payload},
        ]

        return self.backbone.generate_response(messages)
