import json
import re
from typing import List, Dict, Any, Optional, Union
from ..database.neo4j_client import Neo4jClient

class Retriever:
    def __init__(self, backbone: Any, neo4j_client: Neo4jClient):
        self.backbone = backbone
        self.neo4j_client = neo4j_client

    def parse_query(self, query: str, image: Optional[Union[Any, str]] = None) -> Dict[str, List[str]]:
        """
        利用 Qwen 提取 Query 中的实体。
        改进的 prompt：更明确地指导模型提取物体、属性、状态等关键词。
        如果提供 image，会结合主图做更精确的实体提取。
        """
        # 改进的 Prompt：更清晰地指导实体提取
        content_payload = []
        if image is not None:
            content_payload.append({"type": "image", "image": image})

        content_payload.append({"type": "text", "text": (
            "Extract key visual concepts for image retrieval using BOTH the question and the main image (if provided).\n\n"
            "Consider multiple facets (balanced):\n"
            "1) Category/identity (e.g., fruit type, object class).\n"
            "2) Shape/parts/structure (e.g., sliced, cross-section, seeds, stem).\n"
            "3) Visual attributes (color, texture, surface patterns, spots, mold/fuzz, moisture).\n"
            "4) State/condition (fresh/ripe/oxidized/rotting) only if mentioned or visible.\n\n"
            "Guidelines:\n"
            "- Use concrete, observable phrases.\n"
            "- Avoid generic words like 'characteristics', 'feature', 'aspect'.\n"
            "- Keep each concept short (1-3 words) and specific.\n\n"
            "Rules:\n"
            "- 'positive': concepts that SHOULD appear in the target image.\n"
            "- 'negative': concepts explicitly stated as NOT wanted OR explicitly stated as unlikely in the question.\n"
            "  If no negative concepts, use [].\n\n"
            "Output ONLY valid JSON and nothing else.\n"
            "Format: {\"positive\": [...], \"negative\": [...]}\n\n"
            f"Question: {query}\n\n"
            "JSON:"
        )})

        prompt_messages = [
            {
                "role": "user",
                "content": content_payload
            }
        ]
        
        # 调用模型生成
        response_text = self.backbone.generate_response(prompt_messages)
        
        # 解析 JSON
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        try:
            start = clean_text.find("{")
            end = clean_text.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = clean_text[start:end]
                data = json.loads(json_str)
                
                # 确保返回的数据结构正确
                pos = data.get("positive", [])
                neg = data.get("negative", [])
                
                # Fallback: 如果 positive 为空，使用简单关键词提取
                if not pos:
                    pos = self._simple_extract(query)
                    print(f"[Fallback] 简单提取关键词：{pos}")
                
                return {"positive": pos, "negative": neg}
            else:
                print(f"Failed to parse JSON from response: {clean_text}")
                return {"positive": self._simple_extract(query), "negative": []}
        except json.JSONDecodeError:
            print(f"JSON Decode Error: {clean_text}")
            return {"positive": self._simple_extract(query), "negative": []}
    
    def _simple_extract(self, query: str) -> List[str]:
        """
        简单的关键词提取 fallback。
        过滤掉常见的停用词，保留名词和形容词。
        """
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "this", "that", "these", "those",
            "which", "what", "who", "whom", "whose", "where", "when", "why", "how",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "down",
            "out", "off", "over", "under", "again", "further", "then", "once",
            "here", "there", "all", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "just", "and", "but", "if", "or", "because", "as", "until",
            "while", "one", "among", "it", "its", "into", "your", "you", "i", "me",
            "my", "we", "our", "they", "their", "he", "she", "him", "her", "his",
            "about", "after", "before", "during", "between", "through", "above",
            "below", "features", "feature", "following", "likely", "unlikely",
            "characteristic", "characteristics", "aspect", "aspects", "fruit",
            "oxidation", "oxidized", "oxidizes", "undergoes", "post", "undergoing"
        }
        
        # 简单分词，过滤停用词和短词
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # 最多返回 5 个关键词
        return keywords[:5] if keywords else ["object"]

    def sort_and_filter_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 简单排序，实际逻辑已经在 DB 层完成
        return sorted(candidates, key=lambda x: x['score'], reverse=True)

    def retrieve_candidates(self, query: str, limit_img: int = 5, limit_text: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        [修改] 全流程混合检索：解析 -> 图搜索(Image+Text) -> 返回结果
        Returns:
             {
                "images": [...],
                "texts": [...]
             }
        """
        entities = self.parse_query(query)
        pos_concepts = entities.get("positive", [])
        neg_concepts = entities.get("negative", []) 
        
        if not pos_concepts:
            print("No entities extracted from query.")
            return {"images": [], "texts": []}
            
        # 调用新的混合检索接口
        mixed_results = self.neo4j_client.search_mixed_modality(
            pos_concepts, 
            neg_concepts, 
            limit_img=limit_img, 
            limit_text=limit_text
        )
        
        return mixed_results
