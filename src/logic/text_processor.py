import json
import uuid
import re
from typing import List, Dict, Generator
from ..backbone.qwen_wrapper import QwenUnifiedBackbone
from ..database.neo4j_client import Neo4jClient

class TextProcessor:
    def __init__(self, backbone: QwenUnifiedBackbone, neo4j_client: Neo4jClient):
        self.backbone = backbone
        self.neo4j_client = neo4j_client
        self.chunk_size = 512  # tokens estimate (using chars for simplicity ~2000 chars)
        self.overlap = 50      # overlap chars

    def chunk_text(self, text: str) -> List[str]:
        """
        简单的滑动窗口切片。
        更高级的切片可以使用 tiktoken 或 nltk 按句子切分。
        """
        chunks = []
        if not text:
            return chunks
            
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            # Move start pointer back by overlap size for the next chunk
            start = end - self.overlap
            
        return chunks

    def extract_concepts_from_text(self, text_chunk: str) -> List[str]:
        """
        利用 Qwen 提取文本中的关键概念。
        Prompt: "Read the following text and extract key entities, events, and topics strictly as a comma-separated list."
        """
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Read the following text and extract key entities, events, and topics strictly as a comma-separated list. Text: {text_chunk}"}
                ]
            }
        ]
        
        # 调用 Qwen 的 generate_response (需要确保 QwenUnifiedBackbone 有这个通用接口)
        # 如果 QwenUnifiedBackbone 还没有通用 generate 接口，我们需要去添加一个。
        # 假设我们将在 backbone 中添加 `generate_response(messages)`
        response_text = self.backbone.generate_text_response(prompt_messages)
        
        # 清洗结果
        # 预期输出: "Dog, Frisbee, Park, Running"
        concepts = [c.strip() for c in response_text.split(",") if c.strip()]
        
        # 简单的去噪 (去除太长或包含特殊字符的)
        valid_concepts = []
        for c in concepts:
            if len(c) < 50 and re.match(r'^[a-zA-Z0-9\s\-_]+$', c):
                valid_concepts.append(c)
        
        return valid_concepts

    def process_file(self, content: str, source_id: str):
        """
        处理单个文档：切片 -> 提取 -> 入库
        """
        print(f"Processing source: {source_id}...")
        chunks = self.chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            # 1. 生成 Text ID
            text_id = str(uuid.uuid4())
            
            # 2. 存入 Neo4j (Metadata 包含 source 和 chunk index)
            metadata = json.dumps({"source": source_id, "chunk_index": i})
            self.neo4j_client.add_text_node(text_id, chunk, metadata)
            
            # 3. 提取 Concepts
            concepts = self.extract_concepts_from_text(chunk)
            print(f"  Chunk {i}: Extracted {len(concepts)} concepts: {concepts[:3]}...")
            
            # 4. 建立链接
            if concepts:
                self.neo4j_client.add_text_concepts(text_id, concepts)
                
        print(f"Finished processing {source_id}.")
