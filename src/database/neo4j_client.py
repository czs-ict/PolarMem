from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional


class Neo4jClient:
    def __init__(
        self, 
        uri: str = "bolt://localhost:7687", 
        auth: tuple = ("neo4j", "password"),
        dataset: str = "default",
        database: Optional[str] = None
    ):
        self.dataset = dataset
        self.database = database  # Neo4j 数据库名称（用于多数据库隔离）
        try:
            self.driver = GraphDatabase.driver(uri, auth=auth)
            # Verify connection
            self.driver.verify_connectivity()
            
            # 如果指定了 database，尝试创建（如果不存在）
            if self.database:
                self._ensure_database_exists()
            
            db_info = f" (database: {database})" if database else f" (dataset: {dataset})"
            print(f"Connected to Neo4j at {uri}{db_info}")
            self._init_constraints()
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def _ensure_database_exists(self):
        """
        确保指定的数据库存在，如果不存在则创建。
        需要在 system 数据库中执行。
        """
        if not self.driver or not self.database:
            return
        
        try:
            # 在 system 数据库中检查数据库是否存在
            with self.driver.session(database="system") as sys_session:
                # 检查数据库是否存在
                result = sys_session.run(
                    "SHOW DATABASES WHERE name = $name",
                    name=self.database
                )
                
                # 如果数据库不存在，创建它
                if not result.single():
                    # 尝试使用 IF NOT EXISTS 语法（Neo4j 5+）
                    try:
                        sys_session.run(f"CREATE DATABASE {self.database} IF NOT EXISTS")
                        print(f"✓ Created Neo4j database: {self.database}")
                    except Exception as create_error:
                        # 如果 IF NOT EXISTS 不支持，尝试普通 CREATE
                        try:
                            sys_session.run(f"CREATE DATABASE {self.database}")
                            print(f"✓ Created Neo4j database: {self.database}")
                        except Exception:
                            raise create_error
                else:
                    print(f"✓ Neo4j database '{self.database}' already exists")
        except Exception as e:
            error_msg = str(e)
            # 某些 Neo4j 版本可能不支持 SHOW DATABASES 或 CREATE DATABASE
            # 或者权限不足，这里只打印警告，不阻止连接
            print(f"⚠ Warning: Could not check/create database {self.database}: {error_msg}")
            if "UnsupportedAdministrationCommand" in error_msg or "CREATE DATABASE" in error_msg:
                print("  Note: Multi-database feature requires Neo4j Enterprise Edition.")
                print("  For Community Edition, please use dataset attribute isolation instead.")
                print(f"  Or manually create the database using Neo4j Browser/Cypher Shell:")
                print(f"    CREATE DATABASE {self.database}")
            else:
                print("  Please ensure the database exists manually or has proper permissions.")

    def close(self):
        if self.driver:
            self.driver.close()

    def _get_session(self):
        """获取 Neo4j session，如果指定了 database 则使用该数据库。"""
        if self.database:
            return self.driver.session(database=self.database)
        return self.driver.session()
    
    def _init_constraints(self):
        if not self.driver:
            return
        with self._get_session() as session:
            # 确保 Image ID 唯一
            try:
                # Neo4j 4.x/5.x syntax might vary. Using generic CREATE CONSTRAINT
                session.run(
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE"
                )
                # 确保 Concept Name 唯一，避免重复创建节点
                session.run(
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE"
                )
                # [新增] 确保 Text ID 唯一
                session.run(
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Text) REQUIRE t.id IS UNIQUE"
                )
            except Exception as e:
                print(f"Error creating constraints: {e}")

    def add_image_node(self, image_id: str):
        if not self.driver:
            return
        with self._get_session() as session:
            # 先按 id 匹配节点，然后设置 dataset 属性
            # 这样可以处理已存在但没有 dataset 属性的节点
            session.run(
                "MERGE (i:Image {id: $id}) SET i.dataset = $dataset",
                id=image_id,
                dataset=self.dataset
            )

    def add_concepts(
        self, image_id: str, pos_concepts: List[str], neg_concepts: List[str]
    ):
        """
        添加 Concept 节点及 HAS/NOT_HAS 边
        """
        if not self.driver:
            return
        with self._get_session() as session:
            # 处理 Positive Concepts (HAS 边)
            for concept in pos_concepts:
                query = """
                MATCH (i:Image {id: $img_id, dataset: $dataset})
                MERGE (c:Concept {name: $c_name})
                MERGE (i)-[r:HAS]->(c)
                SET r.confidence = 1.0
                """
                session.run(query, img_id=image_id, c_name=concept, dataset=self.dataset)

            # 处理 Negative Concepts (NOT_HAS 边)
            for concept in neg_concepts:
                query = """
                MATCH (i:Image {id: $img_id, dataset: $dataset})
                MERGE (c:Concept {name: $c_name})
                MERGE (i)-[r:NOT_HAS]->(c)
                SET r.confidence = 1.0
                """
                session.run(query, img_id=image_id, c_name=concept, dataset=self.dataset)

    def add_text_node(self, text_id: str, content: str, metadata: str = "{}"):
        """
        [新增] 添加文本节点
        """
        if not self.driver:
            return
        with self._get_session() as session:
            query = """
            MERGE (t:Text {id: $id})
            SET t.content = $content, t.metadata = $metadata
            """
            session.run(query, id=text_id, content=content, metadata=metadata)

    def add_text_concepts(self, text_id: str, concepts: List[str]):
        """
        [新增] 建立 (Text)-[:MENTIONS]->(Concept) 关系
        """
        if not self.driver:
            return
        with self._get_session() as session:
            for concept in concepts:
                query = """
                MATCH (t:Text {id: $tid})
                MERGE (c:Concept {name: $c_name})
                MERGE (t)-[r:MENTIONS]->(c)
                SET r.weight = 1.0 
                """
                session.run(query, tid=text_id, c_name=concept)

    def search_graph(
        self,
        pos_entities: List[str],
        neg_entities: List[str],
        limit: int = 10,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 0.0,
        delta: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        根据 Query 提取的实体进行图谱检索。
        Project Rules 3.2 公式:
        Score(I) = sum(c in Pos) (1.0 * I_HAS) - sum(c in Pos) (2.0 * I_NOT_HAS)

        如果 Query 中的词命中图谱中的 NOT_HAS 边，给予 2.0 倍 负分惩罚。
        """
        if not self.driver or (not pos_entities and not neg_entities):
            return []

        with self._get_session() as session:
            # Weighted scoring (supports ablations by setting weights to 0):
            # pos_entities:
            #   +alpha * (#HAS matches) - beta * (#NOT_HAS matches)
            # neg_entities:
            #   -gamma * (#HAS matches) + delta * (#NOT_HAS matches)
            #
            # Use pattern comprehensions to avoid cartesian products.
            query = """
            MATCH (i:Image {dataset: $dataset})
            WITH i,
                 size([(i)-[:HAS]->(c1:Concept) WHERE c1.name IN $pos_concepts | 1]) AS pos_has,
                 size([(i)-[:NOT_HAS]->(c2:Concept) WHERE c2.name IN $pos_concepts | 1]) AS pos_not,
                 size([(i)-[:HAS]->(c3:Concept) WHERE c3.name IN $neg_concepts | 1]) AS neg_has,
                 size([(i)-[:NOT_HAS]->(c4:Concept) WHERE c4.name IN $neg_concepts | 1]) AS neg_not
            WITH i, pos_has, pos_not, neg_has, neg_not,
                 ($alpha * pos_has) - ($beta * pos_not) - ($gamma * neg_has) + ($delta * neg_not) AS final_score
            WHERE (pos_has + pos_not + neg_has + neg_not) > 0
            RETURN i.id AS image_id, final_score
            ORDER BY final_score DESC
            LIMIT $limit
            """

            try:
                result = session.run(
                    query,
                    pos_concepts=pos_entities or [],
                    neg_concepts=neg_entities or [],
                    limit=limit,
                    dataset=self.dataset,
                    alpha=float(alpha),
                    beta=float(beta),
                    gamma=float(gamma),
                    delta=float(delta),
                )
                candidates = []
                for record in result:
                    candidates.append(
                        {"image_id": record["image_id"], "score": record["final_score"]}
                    )
                return candidates
            except Exception as e:
                print(f"Error searching graph: {e}")
                return []

    def add_similar_relationship(
        self, img_id1: str, img_id2: str, similarity_score: float
    ):
        """
        [新增] 添加图片之间的相似关系
        CREATE (i1)-[:SIMILAR_TO {score: s}]->(i2)
        """
        if not self.driver:
            return
        with self._get_session() as session:
            query = """
            MATCH (i1:Image {id: $id1, dataset: $dataset})
            MATCH (i2:Image {id: $id2, dataset: $dataset})
            MERGE (i1)-[r:SIMILAR_TO]->(i2)
            SET r.score = $score
            """
            session.run(query, id1=img_id1, id2=img_id2, score=similarity_score, dataset=self.dataset)

    def get_similar_images(self, img_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        [新增] 通过图遍历获取与某图片相似的其他图片
        Returns: [{"image_id": "...", "score": 0.85}, ...]
        """
        if not self.driver:
            return []
        with self._get_session() as session:
            query = """
            MATCH (i1:Image {id: $id, dataset: $dataset})-[r:SIMILAR_TO]->(i2:Image {dataset: $dataset})
            RETURN i2.id as image_id, r.score as score
            ORDER BY r.score DESC
            LIMIT $limit
            """
            try:
                result = session.run(query, id=img_id, limit=limit, dataset=self.dataset)
                return [
                    {"image_id": record["image_id"], "score": record["score"]}
                    for record in result
                ]
            except Exception as e:
                print(f"Error getting similar images: {e}")
                return []

    def search_mixed_modality(
        self,
        pos_entities: List[str],
        neg_entities: List[str],
        limit_img: int = 5,
        limit_text: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        [新增] 混合模态检索：同时返回最相关的图片和文本片段

        Returns:
            {
                "images": [{"image_id": "...", "score": 1.5}, ...],
                "texts": [{"text_id": "...", "content": "...", "metadata": "...", "score": 2.0}, ...]
            }
        """
        if not self.driver or not pos_entities:
            return {"images": [], "texts": []}

        with self._get_session() as session:
            # 1. 检索图片 (逻辑同 search_graph)
            img_query = """
            MATCH (c:Concept) WHERE c.name IN $pos_concepts
            MATCH (i:Image)
            OPTIONAL MATCH (i)-[r_has:HAS]->(c)
            OPTIONAL MATCH (i)-[r_not:NOT_HAS]->(c)
            WITH i, c, r_has, r_not
            WHERE r_has IS NOT NULL OR r_not IS NOT NULL
            WITH i, 
                 sum(CASE WHEN r_has IS NOT NULL THEN 1.0 ELSE 0.0 END) as pos_score,
                 sum(CASE WHEN r_not IS NOT NULL THEN 2.0 ELSE 0.0 END) as penalty_score
            WITH i, (pos_score - penalty_score) as final_score
            WHERE final_score > -9999
            RETURN i.id as image_id, final_score
            ORDER BY final_score DESC
            LIMIT $limit
            """

            # 2. 检索文本 (新逻辑)
            # Score(Text) = sum(1.0 * MENTIONS)
            text_query = """
            MATCH (c:Concept) WHERE c.name IN $pos_concepts
            MATCH (t:Text)-[r:MENTIONS]->(c)
            WITH t, sum(r.weight) as score
            RETURN t.id as text_id, t.content as content, t.metadata as metadata, score
            ORDER BY score DESC
            LIMIT $limit
            """

            result_data = {"images": [], "texts": []}

            try:
                # Run Image Query
                img_res = session.run(
                    img_query, pos_concepts=pos_entities, limit=limit_img
                )
                for record in img_res:
                    result_data["images"].append(
                        {"image_id": record["image_id"], "score": record["final_score"]}
                    )

                # Run Text Query
                txt_res = session.run(
                    text_query, pos_concepts=pos_entities, limit=limit_text
                )
                for record in txt_res:
                    result_data["texts"].append(
                        {
                            "id": record["text_id"],
                            "content": record["content"],
                            "metadata": record["metadata"],
                            "score": record["score"],
                        }
                    )

                return result_data
            except Exception as e:
                print(f"Error in mixed modality search: {e}")
                return {"images": [], "texts": []}
