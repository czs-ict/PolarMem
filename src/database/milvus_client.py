from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
from typing import List, Dict, Any
import numpy as np

class MilvusClient:
    def __init__(self, host: str = "localhost", port: str = "19530", collection_name: str = "image_tensors", dim: int = 3584):
        self.collection_name = collection_name
        self.dim = dim
        self.host = host
        self.port = port
        
        # Connect explicitly
        # Note: If connection already exists, we might need to handle it or use alias
        try:
            connections.connect(alias="default", host=host, port=port)
            print(f"Connected to Milvus at {host}:{port}")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            # Depending on setup, might want to raise error or continue if mock is expected
            
        self._init_collection()

    def _init_collection(self):
        # Check if connected before operations
        if not connections.has_connection("default"):
            return 

        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            # 检查已存在 collection 的维度是否匹配
            schema = self.collection.schema
            existing_dim = None
            for field in schema.fields:
                if field.name == "tensor":
                    # 尝试多种方式获取维度
                    # 方式1: field.params dict
                    if hasattr(field, "params") and isinstance(field.params, dict) and "dim" in field.params:
                        existing_dim = field.params["dim"]
                    # 方式2: field.dim 属性
                    elif hasattr(field, "dim"):
                        existing_dim = field.dim
                    # 方式3: 从 type_params 获取（某些 pymilvus 版本）
                    elif hasattr(field, "type_params") and isinstance(field.type_params, dict) and "dim" in field.type_params:
                        existing_dim = int(field.type_params["dim"])
                    break
            
            print(f"Found existing collection {self.collection_name}, existing_dim={existing_dim}, requested_dim={self.dim}")
            
            if existing_dim is not None and existing_dim != self.dim:
                print(f"⚠ Warning: Dimension mismatch! Dropping and recreating collection...")
                utility.drop_collection(self.collection_name)
                # 继续创建新 collection
            elif existing_dim == self.dim:
                # 维度匹配，直接加载
                print(f"Dimension matches, loading existing collection...")
                self.collection.load()
                return
            else:
                # 无法获取维度，为安全起见删除重建
                print(f"⚠ Warning: Cannot determine existing dimension, dropping and recreating...")
                utility.drop_collection(self.collection_name)
        
        # 创建新 collection 或重新创建
        print(f"Creating collection {self.collection_name} with dim={self.dim}...")
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="tensor", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="type", dtype=DataType.INT8), # 0=Global, 1=Patch
            FieldSchema(name="source_img_id", dtype=DataType.VARCHAR, max_length=128),
        ]
        
        schema = CollectionSchema(fields, "Ada-SR-Graph Image Embeddings")
        self.collection = Collection(self.collection_name, schema)
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="tensor", index_params=index_params)
        self.collection.load()

    def insert_embedding(
        self,
        embedding: np.ndarray,
        img_type: int,
        source_img_id: str,
        flush: bool = True,
    ):
        """
        插入单条或多条数据
        embedding: Shape [dim] or [N, dim]
        """
        if self.collection is None:
            print("Collection not initialized.")
            return

        # 归一化输入为 list of lists
        if len(embedding.shape) == 1:
            embeddings = [embedding.tolist()]
            types = [img_type]
            source_ids = [source_img_id]
        else:
            embeddings = embedding.tolist()
            types = [img_type] * len(embeddings)
            source_ids = [source_img_id] * len(embeddings)
            
        data = [
            embeddings,
            types,
            source_ids
        ]
        
        self.collection.insert(data)
        # Flush frequency can be managed by caller for performance.
        if flush:
            self.collection.flush()

    def search_similar(
        self,
        query_tensor: np.ndarray,
        top_k: int = 5,
        expr: str | None = None,
    ) -> List[Dict[str, Any]]:
        if self.collection is None:
            return []
            
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        if len(query_tensor.shape) == 1:
            query_tensor = [query_tensor.tolist()]
        else:
            query_tensor = query_tensor.tolist()
            
        results = self.collection.search(
            data=query_tensor,
            anns_field="tensor",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["source_img_id", "type"]
        )
        
        parsed_results = []
        for hits in results:
            for hit in hits:
                parsed_results.append({
                    "id": hit.id,
                    "distance": hit.distance,
                    "source_img_id": hit.entity.get("source_img_id"),
                    "type": hit.entity.get("type")
                })
                
        return parsed_results

    def get_embedding(self, source_img_id: str, img_type: int = 0) -> np.ndarray | None:
        """Fetch one embedding vector by source_img_id and type.

        Note: Requires collection to exist and be loaded.
        """
        if self.collection is None:
            return None
        try:
            # Query can return float vector field.
            res = self.collection.query(
                expr=f'source_img_id == "{source_img_id}" && type == {int(img_type)}',
                output_fields=["tensor"],
                limit=1,
            )
            if not res:
                return None
            vec = res[0].get("tensor")
            if vec is None:
                return None
            return np.asarray(vec, dtype=np.float32)
        except Exception:
            return None
