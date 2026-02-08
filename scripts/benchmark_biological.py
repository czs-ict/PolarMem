"""MRAG-Bench Biological-only benchmark runner with offline storage and online inference.

This script is specifically for Biological scenario:
- Offline stage: Only process images with "Biological" prefix
- Online stage: Only test samples with scenario="Biological"
- Baseline stage: Only test samples with scenario="Biological"
"""

import argparse
import glob
import io
import os
import subprocess
import sys
from typing import Dict, List, Optional, Any

import torch
from PIL import Image
from tqdm import tqdm
import pyarrow.parquet as pq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backbone.qwen_wrapper import QwenUnifiedBackbone
from src.database.milvus_client import MilvusClient
from src.database.neo4j_client import Neo4jClient
from src.logic.otsu_gate import calculate_dynamic_threshold
from src.logic.patching import OverlapPatcher
from src.pipeline.generator import Generator
from src.pipeline.retriever import Retriever


MRAG_DATA_DIR = os.environ.get("MRAG_DATA_DIR", "data/mrag_bench_raw/data")
MRAG_IMAGE_CORPUS = os.environ.get("MRAG_IMAGE_CORPUS", "data/mrag_bench_image_corpus")


def find_parquet_file(base_dir: str) -> Optional[str]:
    """Find the first parquet file in the directory."""
    parquet_files = sorted(glob.glob(f"{base_dir}/test-*.parquet"))
    if parquet_files:
        return parquet_files[0]
    return None


def load_single_sample(limit: Optional[int] = 1, filter_scenario: Optional[str] = None) -> List[Dict]:
    """Load up to `limit` samples from MRAG-Bench parquet shards.

    遍历 MRAG_DATA_DIR 下所有 test-*.parquet，按文件顺序依次读取，
    累积样本直到达到 limit 或所有数据耗尽。
    
    Args:
        limit: 最大样本数
        filter_scenario: 如果指定，只加载 scenario 匹配的样本
    """
    parquet_files = sorted(glob.glob(f"{MRAG_DATA_DIR}/test-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet file found in {MRAG_DATA_DIR}")

    samples: List[Dict] = []
    max_samples = float("inf") if limit is None else int(limit)

    for parquet_path in parquet_files:
        if len(samples) >= max_samples:
            break

        table = pq.read_table(parquet_path)
        num_rows = len(table)
        if num_rows == 0:
            continue

        # 读取所有行，然后过滤
        for i in range(num_rows):
            if len(samples) >= max_samples:
                break
                
            row = table.slice(i, 1).to_pydict()
            scenario = row.get("scenario", [None])[0]
            
            # 如果指定了 filter_scenario，只加载匹配的样本
            if filter_scenario and scenario != filter_scenario:
                continue
            
            sample: Dict[str, Any] = {
                "id": row["id"][0],
                "scenario": scenario,
                "aspect": row["aspect"][0],
                "question": row["question"][0],
                "options": {
                    "A": row["A"][0],
                    "B": row["B"][0],
                    "C": row["C"][0],
                    "D": row["D"][0],
                },
                "answer_choice": row["answer_choice"][0],
                "answer": row["answer"][0],
                "image_type": row["image_type"][0],
            }
            if "image" in row and len(row["image"]) > 0:
                img_data = row["image"][0]
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img_bytes = img_data["bytes"]
                    if img_bytes:
                        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        sample["image"] = image

            samples.append(sample)

    return samples


def load_image_by_id(image_id: str) -> Optional[Image.Image]:
    """Load image from corpus by ID."""
    for ext in [".jpg", ".png", ".jpeg"]:
        img_path = os.path.join(MRAG_IMAGE_CORPUS, f"{image_id}{ext}")
        if os.path.exists(img_path):
            return Image.open(img_path).convert("RGB")
    return None


def _list_image_files(image_dir: str, prefix_filter: Optional[str] = None) -> List[str]:
    """List image files in a directory.
    
    Args:
        image_dir: 图片目录
        prefix_filter: 如果指定，只返回文件名以此前缀开头的文件
    """
    all_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    
    if prefix_filter:
        all_files = [f for f in all_files if f.startswith(prefix_filter)]
    
    return all_files


def _shard_items(items: List[str], rank: int, world_size: int) -> List[str]:
    """Deterministically shard a list by index modulo world_size."""
    if world_size <= 1:
        return items
    return [x for idx, x in enumerate(items) if (idx % world_size) == rank]


def offline_store(
    image_dir: str,
    similarity_threshold: float = 0.7,
    limit: Optional[int] = None,
    dataset: str = "biological",
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_database: Optional[str] = None,
    milvus_host: str = "localhost",
    milvus_port: str = "19530",
    worker_rank: int = 0,
    worker_world_size: int = 1,
    skip_relationships: bool = False,
    reset_neo4j_dataset: bool = False,
    reset_milvus_collection: bool = False,
    milvus_flush_every: int = 2000,
    skip_embeddings: bool = False,
    skip_milvus_insert: bool = False,
) -> None:
    """Offline storage: build concept graph, embeddings, and relationships.
    
    只处理文件名前缀为 "Biological" 的图片。

    Args:
        image_dir: Directory containing images.
        similarity_threshold: Minimum similarity score for SIMILAR_TO.
        limit: Optional limit on number of images for quick tests.
        dataset: Dataset name for database isolation (default: "biological").
    """
    print("=" * 60)
    print("离线阶段：数据存储与索引构建（仅 Biological 图片）")
    print("=" * 60)

    if not os.path.exists(image_dir):
        print(f"✗ 图片目录不存在：{image_dir}")
        return

    print("正在初始化组件...")
    try:
        backbone = QwenUnifiedBackbone()
        # 使用 dataset 名称作为 collection 后缀，实现 Milvus 隔离
        collection_name = f"image_tensors_{dataset}" if dataset != "default" else "image_tensors"
        if reset_milvus_collection:
            try:
                from pymilvus import connections, utility

                connections.connect(alias="default", host=milvus_host, port=milvus_port)
                if utility.has_collection(collection_name):
                    print(f"⚠ 正在删除 Milvus collection: {collection_name}")
                    utility.drop_collection(collection_name)
            except Exception as e:
                print(f"⚠ 删除 Milvus collection 失败（忽略）：{e}")

        milvus = None
        if not skip_milvus_insert and not skip_embeddings:
            milvus = MilvusClient(
                host=milvus_host,
                port=milvus_port,
                collection_name=collection_name
            )
        # 如果指定了 neo4j_database，使用不同的数据库实例；否则使用 dataset 属性隔离
        neo4j = Neo4jClient(
            uri=neo4j_uri,
            dataset=dataset if not neo4j_database else "default",
            database=neo4j_database
        )
        if reset_neo4j_dataset and neo4j.driver is not None:
            try:
                with neo4j._get_session() as session:  # noqa: SLF001 (script-level internal access)
                    session.run(
                        "MATCH (i:Image {dataset:$dataset}) DETACH DELETE i",
                        dataset=dataset,
                    )
                print(f"⚠ 已清理 Neo4j 中 dataset='{dataset}' 的 Image 节点")
            except Exception as e:
                print(f"⚠ 清理 Neo4j dataset 失败（忽略）：{e}")

        patcher = OverlapPatcher()
    except Exception as e:
        print(f"✗ 初始化失败：{e}")
        return

    # 只获取前缀为 "Biological" 的图片
    image_files = sorted(_list_image_files(image_dir, prefix_filter="Biological"))
    # 兼容旧用法：--limit 0 表示“不限制”（处理全部）
    if limit is not None:
        try:
            limit_int = int(limit)
        except Exception:
            limit_int = None
        if limit_int is not None and limit_int > 0:
            image_files = image_files[:limit_int]

    # 多卡并行时，每个 worker 只处理自己的 shard
    if worker_world_size > 1:
        image_files = _shard_items(image_files, worker_rank, worker_world_size)
        print(
            f"并行建库：worker {worker_rank}/{worker_world_size} "
            f"将处理 {len(image_files)} 张图片"
        )

    print(f"共发现 {len(image_files)} 张 Biological 图片，开始离线存储...")

    embeddings_cache: Dict[str, object] = {}
    milvus_insert_count = 0

    for img_file in tqdm(image_files, desc="离线存储进度"):
        img_path = os.path.join(image_dir, img_file)
        img_id = os.path.splitext(img_file)[0]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"加载图片失败：{img_file}，原因：{e}")
            continue

        try:
            neo4j.add_image_node(img_id)
        except Exception as e:
            print(f"写入 Neo4j 失败：{img_id}，原因：{e}")

        if not skip_embeddings:
            # Global embedding
            try:
                global_emb = backbone.extract_embeddings(image)
                embeddings_cache[img_id] = global_emb.float().cpu().numpy()
                if milvus is not None and not skip_milvus_insert:
                    milvus.insert_embedding(
                        embeddings_cache[img_id],
                        img_type=0,
                        source_img_id=img_id,
                        flush=False,
                    )
                    milvus_insert_count += 1
            except Exception as e:
                print(f"全局向量写入失败：{img_id}，原因：{e}")

            # Patch embeddings
            try:
                patches = patcher.process(image)
                for patch_img, _coords in patches:
                    patch_emb = backbone.extract_embeddings(patch_img)
                    if milvus is not None and not skip_milvus_insert:
                        milvus.insert_embedding(
                            patch_emb.float().cpu().numpy(),
                            img_type=1,
                            source_img_id=img_id,
                            flush=False,
                        )
                        milvus_insert_count += 1
            except Exception as e:
                print(f"Patch 向量写入失败：{img_id}，原因：{e}")

        # Periodic Milvus flush for better durability without killing throughput
        if (
            milvus is not None
            and milvus_flush_every > 0
            and milvus_insert_count >= milvus_flush_every
        ):
            try:
                milvus.collection.flush()
            except Exception:
                pass
            milvus_insert_count = 0

        # Concepts + Otsu gate
        try:
            # caption_prompt = [
            #     {
            #         "role": "user",
            #         "content": [
            #             {"type": "image", "image": img_path},
            #             {
            #                 "type": "text",
            #                 "text": (
            #                     "List all visible objects and attributes in this image "
            #                     "strictly as a comma-separated list."
            #                 ),
            #             },
            #         ],
            #     }
            # ]
            caption_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {
                            "type": "text",
                            "text": (
                                "List key visual concepts for retrieval as a comma-separated list. "
                                "Cover multiple facets: category/identity, shape/parts/structure, "
                                "visual attributes (color/texture/spots/mold), and state/condition "
                                "if visible. Avoid generic words like 'characteristics' or 'feature'."
                            ),
                        },
                    ],
                }
            ]
            caption = backbone.generate_response(caption_prompt)
            concepts = [c.strip() for c in caption.split(",") if c.strip()]
            if concepts:
                # 分批处理 VQA 打分，避免显存溢出
                batch_size = 10  # 每批最多 10 个 concepts
                all_scores = {}
                for i in range(0, len(concepts), batch_size):
                    batch_concepts = concepts[i:i+batch_size]
                    batch_scores = backbone.get_vqa_scores(image, batch_concepts)
                    all_scores.update(batch_scores)
                    # 每批后清理显存
                    torch.cuda.empty_cache()
                
                pos, neg = calculate_dynamic_threshold(all_scores)
                neo4j.add_concepts(img_id, pos, neg)
        except Exception as e:
            print(f"概念抽取失败：{img_id}，原因：{e}")
        
        # 每张图片处理完后清理显存缓存
        torch.cuda.empty_cache()

    # Final flush after inserts
    if milvus is not None:
        try:
            milvus.collection.flush()
        except Exception:
            pass

    if not skip_relationships and worker_world_size <= 1:
        print("\n开始构建 SIMILAR_TO 关系...")
        img_ids = list(embeddings_cache.keys())

        for img_id in tqdm(img_ids, desc="关系构建进度"):
            emb = embeddings_cache[img_id]
            try:
                if milvus is None:
                    continue
                results = milvus.search_similar(emb, top_k=5, expr="type == 0")
                for r in results:
                    similar_id = r.get("source_img_id")
                    similarity = r.get("distance", 0)
                    if (
                        similar_id
                        and similar_id != img_id
                        and similarity >= similarity_threshold
                    ):
                        neo4j.add_similar_relationship(img_id, similar_id, similarity)
            except Exception as e:
                print(f"关系构建失败：{img_id}，原因：{e}")

    if hasattr(neo4j, "close"):
        neo4j.close()

    print("\n✓ 离线存储完成")


def build_relationships(
    dataset: str = "biological",
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_database: Optional[str] = None,
    milvus_host: str = "localhost",
    milvus_port: str = "19530",
    similarity_threshold: float = 0.7,
    top_k: int = 5,
    reset_existing: bool = True,
) -> None:
    """Build SIMILAR_TO edges after embeddings are already in Milvus."""
    print("=" * 60)
    print("构建 SIMILAR_TO 关系（基于 Milvus 全局向量）")
    print("=" * 60)

    # DB clients
    collection_name = f"image_tensors_{dataset}" if dataset != "default" else "image_tensors"
    milvus = MilvusClient(host=milvus_host, port=milvus_port, collection_name=collection_name)
    neo4j = Neo4jClient(
        uri=neo4j_uri,
        dataset=dataset if not neo4j_database else "default",
        database=neo4j_database,
    )
    if neo4j.driver is None or milvus.collection is None:
        print("✗ 数据库连接失败，无法构建关系")
        return

    if reset_existing:
        try:
            with neo4j._get_session() as session:  # noqa: SLF001 (script-level internal access)
                session.run(
                    "MATCH (:Image {dataset:$dataset})-[r:SIMILAR_TO]->(:Image {dataset:$dataset}) "
                    "DELETE r",
                    dataset=dataset,
                )
            print("⚠ 已清理旧的 SIMILAR_TO 关系")
        except Exception as e:
            print(f"⚠ 清理旧关系失败（忽略）：{e}")

    # Get all image ids from Neo4j
    with neo4j._get_session() as session:  # noqa: SLF001
        rows = session.run("MATCH (i:Image {dataset:$dataset}) RETURN i.id AS id", dataset=dataset)
        img_ids = [r["id"] for r in rows]

    if not img_ids:
        print(f"✗ Neo4j 中 dataset='{dataset}' 没有 Image，无法构建关系")
        return

    for img_id in tqdm(img_ids, desc="关系构建进度"):
        emb = milvus.get_embedding(img_id, img_type=0)
        if emb is None:
            continue
        try:
            results = milvus.search_similar(emb, top_k=top_k, expr="type == 0")
            for r in results:
                similar_id = r.get("source_img_id")
                similarity = r.get("distance", 0)
                if (
                    similar_id
                    and similar_id != img_id
                    and similarity >= similarity_threshold
                ):
                    neo4j.add_similar_relationship(img_id, similar_id, float(similarity))
        except Exception:
            continue

    neo4j.close()
    print("\n✓ SIMILAR_TO 关系构建完成")


def offline_parallel(
    image_dir: str,
    similarity_threshold: float,
    limit: Optional[int],
    dataset: str,
    neo4j_uri: str,
    neo4j_database: Optional[str],
    milvus_host: str,
    milvus_port: str,
    gpus: str,
    reset_neo4j_dataset: bool,
    reset_milvus_collection: bool,
    milvus_flush_every: int,
    skip_embeddings: bool,
    skip_milvus_insert: bool,
) -> None:
    """Spawn multiple GPU workers to build index faster."""
    gpu_list = [x.strip() for x in gpus.split(",") if x.strip()]
    if not gpu_list:
        raise ValueError("gpus is empty, example: --gpus 4,5,6,7")

    world_size = len(gpu_list)
    print("=" * 60)
    print(f"并行离线建库：使用 GPUs={gpu_list} (world_size={world_size})")
    print("=" * 60)

    # Only reset once in main process to avoid race (no model loading here).
    if reset_milvus_collection:
        collection_name = f"image_tensors_{dataset}" if dataset != "default" else "image_tensors"
        try:
            from pymilvus import connections, utility

            connections.connect(alias="default", host=milvus_host, port=milvus_port)
            if utility.has_collection(collection_name):
                print(f"⚠ 正在删除 Milvus collection: {collection_name}")
                utility.drop_collection(collection_name)
        except Exception as e:
            print(f"⚠ 删除 Milvus collection 失败（忽略）：{e}")

    if reset_neo4j_dataset:
        try:
            neo4j = Neo4jClient(
                uri=neo4j_uri,
                dataset=dataset if not neo4j_database else "default",
                database=neo4j_database,
            )
            if neo4j.driver is not None:
                with neo4j._get_session() as session:  # noqa: SLF001
                    session.run(
                        "MATCH (i:Image {dataset:$dataset}) DETACH DELETE i",
                        dataset=dataset,
                    )
                print(f"⚠ 已清理 Neo4j 中 dataset='{dataset}' 的 Image 节点")
            neo4j.close()
        except Exception as e:
            print(f"⚠ 清理 Neo4j dataset 失败（忽略）：{e}")

    procs: List[subprocess.Popen] = []
    for rank, gpu in enumerate(gpu_list):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "offline",
            "--image_dir", image_dir,
            "--similarity_threshold", str(similarity_threshold),
            "--limit", str(limit if limit is not None else 0),
            "--dataset", dataset,
            "--neo4j_uri", neo4j_uri,
            "--milvus_host", milvus_host,
            "--milvus_port", str(milvus_port),
            "--worker_rank", str(rank),
            "--worker_world_size", str(world_size),
            "--skip_relationships",
            "--milvus_flush_every", str(milvus_flush_every),
        ]
        if skip_embeddings:
            cmd.append("--skip_embeddings")
        if skip_milvus_insert:
            cmd.append("--skip_milvus_insert")
        if neo4j_database:
            cmd.extend(["--neo4j_database", neo4j_database])
        print(f"启动 worker {rank}/{world_size} on GPU {gpu}")
        procs.append(subprocess.Popen(cmd, env=env))

    exit_codes = [p.wait() for p in procs]
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"Some workers failed: {exit_codes}")

    # Build SIMILAR_TO edges after all embeddings exist.
    build_relationships(
        dataset=dataset,
        neo4j_uri=neo4j_uri,
        neo4j_database=neo4j_database,
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        similarity_threshold=similarity_threshold,
        top_k=5,
        reset_existing=True,
    )


def online_infer(
    limit: int = 1,
    dataset: str = "biological",
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_database: Optional[str] = None,
    milvus_host: str = "localhost",
    milvus_port: str = "19530",
    run_name: str = "QWEN_RAG_FULL",
    out_dir: str = "results_bio_ablation/Qwen25VL7B",
    rag_mode: str = "ours_full",
    top_images: int = 5,
    # Weighted ablations (used in rag_mode="ours")
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 0.0,
    delta: float = 0.0,
    lambda_conceptset: float = 1.0,
    lambda_qimg: float = 1.0,
) -> None:
    """Online inference: retrieval + RAG generation using stored data.
    
    只测试 scenario="Biological" 的样本。
    """
    print("=" * 60)
    print("在线阶段：检索增强推理（仅 Biological 场景）")
    print("=" * 60)

    try:
        # 只加载 scenario="Biological" 的样本
        samples = load_single_sample(limit, filter_scenario="Biological")
    except Exception as e:
        print(f"✗ 加载样本失败：{e}")
        return

    if not samples:
        print("✗ 未加载到任何 Biological 样本")
        return

    print(f"已加载 {len(samples)} 个 Biological 样本")

    print("正在初始化组件...")
    try:
        backbone = QwenUnifiedBackbone()
        generator = Generator(backbone)
    except Exception as e:
        print(f"✗ 初始化失败：{e}")
        return

    # Initialize DB clients (do not build index here)
    print("正在连接数据库...")
    rag_mode = rag_mode.lower()
    neo4j: Optional[Neo4jClient] = None
    milvus: Optional[MilvusClient] = None
    neo4j_available = False
    milvus_available = False

    try:
        # 如果指定了 neo4j_database，使用不同的数据库实例；否则使用 dataset 属性隔离
        neo4j = Neo4jClient(
            uri=neo4j_uri,
            dataset=dataset if not neo4j_database else "default",
            database=neo4j_database
        )
        if neo4j.driver is not None:
            neo4j_available = True
    except Exception as e:
        print(f"Neo4j 连接失败：{e}")

    try:
        # 使用 dataset 名称作为 collection 后缀，实现 Milvus 隔离
        collection_name = f"image_tensors_{dataset}" if dataset != "default" else "image_tensors"
        milvus = MilvusClient(
            host=milvus_host,
            port=milvus_port,
            collection_name=collection_name
        )
        if hasattr(milvus, "collection") and milvus.collection is not None:
            milvus_available = True
    except Exception as e:
        print(f"Milvus 连接失败：{e}")

    if not neo4j_available and not milvus_available:
        print("⚠ 未检测到数据库连接，将仅测试推理流程")
    elif neo4j_available and neo4j and neo4j.driver is not None:
        # 快速自检：当前 Neo4j 中是否存在指定 dataset 的 Image 节点
        try:
            with neo4j._get_session() as session:  # noqa: SLF001 (script-level internal access)
                record = session.run(
                    "MATCH (i:Image {dataset: $dataset}) RETURN count(i) AS n",
                    dataset=dataset,
                ).single()
                # neo4j.Record supports record["n"]; using `"n" in record` is unreliable and may
                # cause false 0 counts, so only guard on record existence.
                n_imgs = int(record["n"]) if record is not None else 0
            if n_imgs == 0:
                print(
                    f"⚠ Neo4j 中 dataset='{dataset}' 的图片数量为 0，"
                    "图检索将始终返回 0；将自动尝试退化为 Milvus 向量检索。"
                )
        except Exception as e:
            print(f"⚠ Neo4j 数据自检失败（忽略，不影响继续运行）：{e}")

    # 用于统计的结果列表
    results: List[Dict[str, Any]] = []

    for i, sample in enumerate(samples, start=1):
        print("\n" + "-" * 60)
        print(f"样本 {i}")

        if "image" not in sample:
            print("✗ 样本无图像数据，跳过")
            continue

        question = sample["question"]
        options = [
            sample["options"]["A"],
            sample["options"]["B"],
            sample["options"]["C"],
            sample["options"]["D"],
        ]
        ground_truth = sample["answer_choice"]
        print(f"问题：{question}")
        print(f"选项：A) {options[0]}  B) {options[1]}")
        print(f"      C) {options[2]}  D) {options[3]}")

        image = sample["image"]
        embedding_tensor = backbone.extract_embeddings(image)
        query_embedding = embedding_tensor.float().cpu().numpy()

        # Parse query to extract entities
        entities = {"positive": [], "negative": []}
        if neo4j_available and neo4j and rag_mode in ("ours_full", "ours", "graph_only"):
            retriever = Retriever(backbone, neo4j)
            entities = retriever.parse_query(question)
            print(f"解析实体：{entities}")
        else:
            entities["positive"] = question.lower().split()[:5]
            print(f"简化实体：{entities['positive']}")

        # Graph-based / embedding-based retrieval
        retrieved_images: List[str] = []
        if rag_mode == "ours":
            # OURS (fusion): graph score + qimg similarity
            graph_results: List[Dict[str, Any]] = []
            qimg_results: List[Dict[str, Any]] = []
            if neo4j_available and neo4j:
                graph_results = neo4j.search_graph(
                    entities.get("positive", []),
                    entities.get("negative", []),
                    limit=max(top_images * 5, top_images),
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    delta=delta,
                )
            if milvus_available and milvus:
                qimg_results = milvus.search_similar(
                    query_embedding,
                    top_k=max(top_images * 5, top_images),
                    expr="type == 0",
                )

            # Normalize graph scores into [0,1]
            graph_scores: Dict[str, float] = {}
            for r in graph_results:
                if r.get("image_id") is not None:
                    graph_scores[str(r["image_id"])] = float(r.get("score", 0.0))

            graph_norm: Dict[str, float] = {}
            if graph_scores:
                vals = list(graph_scores.values())
                vmin, vmax = min(vals), max(vals)
                if vmax > vmin:
                    graph_norm = {k: (v - vmin) / (vmax - vmin) for k, v in graph_scores.items()}
                else:
                    graph_norm = {k: 1.0 for k in graph_scores.keys()}

            # QIMG similarity is already in [0,1] for COSINE in this setup
            qimg_norm: Dict[str, float] = {}
            for r in qimg_results:
                sid = r.get("source_img_id")
                if not sid:
                    continue
                sim = float(r.get("distance", 0.0))
                if sim < 0:
                    sim = 0.0
                if sim > 1:
                    sim = 1.0
                # Keep best similarity per image id
                qimg_norm[sid] = max(qimg_norm.get(sid, 0.0), sim)

            # Fuse
            fused: Dict[str, float] = {}
            for k, v in graph_norm.items():
                fused[k] = fused.get(k, 0.0) + (lambda_conceptset * v)
            for k, v in qimg_norm.items():
                fused[k] = fused.get(k, 0.0) + (lambda_qimg * v)

            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
            retrieved_images = [k for k, _ in ranked[:top_images]]

            print(f"图检索结果：{len(graph_results)}")
            print(f"向量检索结果：{len(qimg_results)}")
            print(f"融合检索结果：{len(retrieved_images)}")
        else:
            # Existing ablations:
            # - ours_full: graph-first, fallback to qimg when empty
            # - graph_only: graph only
            # - qimg_only: qimg only
            if neo4j_available and neo4j and rag_mode in ("ours_full", "graph_only"):
                graph_results = neo4j.search_graph(
                    entities.get("positive", []),
                    entities.get("negative", []),
                    limit=top_images,
                )
                print(f"图检索结果：{len(graph_results)}")
                if graph_results:
                    retrieved_images = [r["image_id"] for r in graph_results if r.get("image_id")]
                elif rag_mode == "ours_full" and milvus_available and milvus:
                    search_results = milvus.search_similar(query_embedding, top_k=top_images, expr="type == 0")
                    retrieved_images = [r["source_img_id"] for r in search_results if r.get("source_img_id")]
                    print(f"向量检索结果：{len(retrieved_images)}")
            elif milvus_available and milvus and rag_mode == "qimg_only":
                search_results = milvus.search_similar(query_embedding, top_k=top_images, expr="type == 0")
                retrieved_images = [r["source_img_id"] for r in search_results if r.get("source_img_id")]

        # Relationship traversal（仅在使用图检索时启用）
        all_related_images: Dict[str, float] = {}
        if neo4j_available and neo4j and rag_mode in ("ours_full", "ours", "graph_only"):
            for img_id in retrieved_images:
                similar = neo4j.get_similar_images(img_id, limit=2)
                if similar:
                    for s in similar:
                        all_related_images[s["image_id"]] = s.get("score", 0)

        # Load images
        all_images: Dict[str, Image.Image] = {"main": image}
        for img_id in retrieved_images[:top_images]:
            img = load_image_by_id(img_id)
            if img:
                all_images[f"retrieved_{img_id}"] = img

        for img_id, rel_score in list(all_related_images.items())[: top_images]:
            key = f"related_{img_id}"
            if key not in all_images:
                img = load_image_by_id(img_id)
                if img:
                    all_images[key] = img

        relationship_context = ""
        if neo4j_available and neo4j and rag_mode in ("ours_full", "ours", "graph_only"):
            for img_id in retrieved_images[:top_images]:
                similar = neo4j.get_similar_images(img_id, limit=2)
                if similar:
                    related_ids = [s["image_id"] for s in similar]
                    relationship_context += (
                        f"- {img_id} is similar to: {', '.join(related_ids)}\n"
                    )

        images_for_gen = {k: v for k, v in all_images.items() if k != "main"}
        if relationship_context:
            query_with_context = (
                "You are answering a multiple choice question using RETRIEVAL-AUGMENTED reasoning.\n\n"
                "## Image Structure:\n"
                "- MAIN IMAGE: The question image that you need to analyze.\n"
                "- RETRIEVED IMAGES: Similar/related images retrieved from a knowledge base. These images contain relevant visual patterns, concepts, or examples that can help you better understand the main image and answer the question.\n\n"
                "## Image Relationships:\n"
                f"{relationship_context}\n\n"
                f"## Question: {question}\n\n"
                "## Options:\n"
                f"A) {options[0]}\n"
                f"B) {options[1]}\n"
                f"C) {options[2]}\n"
                f"D) {options[3]}\n\n"
                "## Instructions (RETRIEVAL-AUGMENTED):\n"
                "1. First, carefully examine the MAIN IMAGE to understand what it shows.\n"
                "2. Then, analyze the RETRIEVED IMAGES to identify relevant visual patterns, concepts, or similar cases that relate to the question.\n"
                "3. Use the retrieved images to enhance your understanding of the main image - they may show similar objects, states, transformations, or provide context that helps interpret the main image.\n"
                "4. Consider the relationships between images (similarity links) to understand how they connect to the main image.\n"
                "5. Synthesize information from both the main image and retrieved images to determine the MOST LIKELY correct answer.\n"
                "6. Do NOT express uncertainty, doubt, or provide explanations.\n"
                "7. Output ONLY the answer in the exact format below.\n\n"
                "## Output Format (STRICT):\n"
                "Answer: X\n"
                "where X is one of A, B, C, or D.\n\n"
                "CRITICAL: Do NOT include any text, explanation, reasoning, or uncertainty after 'Answer: X'. Just the answer."
            )
        else:
            query_with_context = (
                "You are answering a multiple choice question using RETRIEVAL-AUGMENTED reasoning.\n\n"
                "## Image Structure:\n"
                "- MAIN IMAGE: The question image that you need to analyze.\n"
                "- RETRIEVED IMAGES: Similar/related images retrieved from a knowledge base. These images contain relevant visual patterns, concepts, or examples that can help you better understand the main image and answer the question.\n\n"
                f"## Question: {question}\n\n"
                "## Options:\n"
                f"A) {options[0]}\n"
                f"B) {options[1]}\n"
                f"C) {options[2]}\n"
                f"D) {options[3]}\n\n"
                "## Instructions (RETRIEVAL-AUGMENTED):\n"
                "1. First, carefully examine the MAIN IMAGE to understand what it shows.\n"
                "2. Then, analyze the RETRIEVED IMAGES to identify relevant visual patterns, concepts, or similar cases that relate to the question.\n"
                "3. Use the retrieved images to enhance your understanding of the main image - they may show similar objects, states, transformations, or provide context that helps interpret the main image.\n"
                "4. Synthesize information from both the main image and retrieved images to determine the MOST LIKELY correct answer.\n"
                "5. Do NOT express uncertainty, doubt, or provide explanations.\n"
                "6. Output ONLY the answer in the exact format below.\n\n"
                "## Output Format (STRICT):\n"
                "Answer: X\n"
                "where X is one of A, B, C, or D.\n\n"
                "CRITICAL: Do NOT include any text, explanation, reasoning, or uncertainty after 'Answer: X'. Just the answer."
            )

        response = generator.generate_with_relationships(
            main_image=image,
            patches=[],
            query=query_with_context,
            retrieved_images=images_for_gen,
            relationship_context=relationship_context,
        )
        # 生成后清理显存缓存（避免长跑时显存碎片堆积）
        torch.cuda.empty_cache()

        import re

        match = re.search(r"Answer:\s*([A-D])", response, re.IGNORECASE)
        if match:
            predicted = match.group(1).upper()
            is_correct = predicted == ground_truth
            print(f"预测：{predicted}，真实：{ground_truth}，匹配：{is_correct}")

            # 记录结果
            results.append(
                {
                    "sample_id": sample.get("id", f"sample_{i}"),
                    "scenario": sample.get("scenario", "unknown"),
                    "aspect": sample.get("aspect", "unknown"),
                    "predicted": predicted,
                    "ground_truth": ground_truth,
                    "correct": is_correct,
                    "question": question,
                }
            )
        else:
            print("✗ 无法解析答案")
            # 记录无法解析的情况
            results.append(
                {
                    "sample_id": sample.get("id", f"sample_{i}"),
                    "scenario": sample.get("scenario", "unknown"),
                    "aspect": sample.get("aspect", "unknown"),
                    "predicted": None,
                    "ground_truth": ground_truth,
                    "correct": False,
                    "question": question,
                }
            )

    if neo4j and hasattr(neo4j, "close"):
        neo4j.close()

    # 统计结果
    print("\n" + "=" * 60)
    print("测试结果统计（Biological 场景）")
    print("=" * 60)
    
    total_samples = len(results)
    valid_predictions = sum(1 for r in results if r["predicted"] is not None)
    correct_predictions = sum(1 for r in results if r.get("correct", False))
    
    print(f"总样本数：{total_samples}")
    print(f"有效预测数：{valid_predictions} (无法解析答案：{total_samples - valid_predictions})")
    print(f"正确数：{correct_predictions}")
    
    if valid_predictions > 0:
        accuracy = correct_predictions / valid_predictions
        print(f"准确率：{accuracy:.4f} ({accuracy * 100:.2f}%)")
    else:
        print("准确率：N/A (无有效预测)")
    
    # 按 aspect 统计
    aspect_stats: Dict[str, Dict[str, int]] = {}
    for r in results:
        aspect = r["aspect"]
        if aspect not in aspect_stats:
            aspect_stats[aspect] = {"total": 0, "correct": 0, "valid": 0}
        aspect_stats[aspect]["total"] += 1
        if r["predicted"] is not None:
            aspect_stats[aspect]["valid"] += 1
            if r.get("correct", False):
                aspect_stats[aspect]["correct"] += 1
    
    if aspect_stats:
        print("\n按 Aspect 统计：")
        print("-" * 60)
        print(f"{'Aspect':<30} {'准确率':<15} {'正确数/有效数/总数'}")
        print("-" * 60)
        for aspect, stats in sorted(aspect_stats.items()):
            if stats["valid"] > 0:
                acc = stats["correct"] / stats["valid"]
                print(
                    f"{aspect:<30} {acc:.4f} ({acc*100:.2f}%)  "
                    f"{stats['correct']}/{stats['valid']}/{stats['total']}"
                )
            else:
                print(
                    f"{aspect:<30} N/A (无有效预测)  "
                    f"{stats['correct']}/{stats['valid']}/{stats['total']}"
                )

    print("=" * 60)

    # 写出 MRAG-Bench 兼容的 jsonl 结果，便于统一打分
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{run_name}.jsonl")
        import json

        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                record = {
                    "id": r["sample_id"],
                    "scenario": r["scenario"],
                    "aspect": r["aspect"],
                    "gt_choice": r["ground_truth"],
                    # 与 score.py 约定：从 output 字段中解析最终答案
                    "output": (
                        f"Answer: {r['predicted']}"
                        if r["predicted"] is not None
                        else ""
                    ),
                    # 作为 prompt 传入，便于 score.py 在需要时做额外解析
                    "prompt": r["question"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"✓ 已写入预测结果到 {out_path}")
    except Exception as e:
        print(f"⚠ 写入预测结果失败：{e}")

    print("\n✓ 在线推理完成")


def online_infer_baseline(
    limit: int = 1,
    run_name: str = "QWEN_NO_RAG",
    out_dir: str = "results_bio_ablation/Qwen25VL7B",
) -> None:
    """Baseline inference: no retrieval enhancement, only use main image.
    
    只测试 scenario="Biological" 的样本。
    """
    print("=" * 60)
    print("基线测试：原生模型性能（无检索增强，仅 Biological 场景）")
    print("=" * 60)

    try:
        # 只加载 scenario="Biological" 的样本
        samples = load_single_sample(limit, filter_scenario="Biological")
    except Exception as e:
        print(f"✗ 加载样本失败：{e}")
        return

    if not samples:
        print("✗ 未加载到任何 Biological 样本")
        return

    print(f"已加载 {len(samples)} 个 Biological 样本")

    print("正在初始化组件...")
    try:
        backbone = QwenUnifiedBackbone()
        generator = Generator(backbone)
    except Exception as e:
        print(f"✗ 初始化失败：{e}")
        return

    # 用于统计的结果列表
    results: List[Dict[str, Any]] = []

    for i, sample in enumerate(samples, start=1):
        print("\n" + "-" * 60)
        print(f"样本 {i}")

        if "image" not in sample:
            print("✗ 样本无图像数据，跳过")
            continue

        question = sample["question"]
        options = [
            sample["options"]["A"],
            sample["options"]["B"],
            sample["options"]["C"],
            sample["options"]["D"],
        ]
        ground_truth = sample["answer_choice"]
        print(f"问题：{question}")
        print(f"选项：A) {options[0]}  B) {options[1]}")
        print(f"      C) {options[2]}  D) {options[3]}")

        image = sample["image"]

        # 基线版本：只使用主图，不进行检索
        # 构造简化的 prompt（不包含检索图片和关系信息）
        query_with_context = (
            "You are answering a multiple choice question about the MAIN IMAGE.\n\n"
            f"## Question: {question}\n\n"
            "## Options:\n"
            f"A) {options[0]}\n"
            f"B) {options[1]}\n"
            f"C) {options[2]}\n"
            f"D) {options[3]}\n\n"
            "## Instructions:\n"
            "1. Analyze the MAIN IMAGE carefully to answer the question.\n"
            "2. You MUST choose the MOST LIKELY correct option based on the main image.\n"
            "3. Do NOT express uncertainty, doubt, or provide explanations.\n"
            "4. Output ONLY the answer in the exact format below.\n\n"
            "## Output Format (STRICT):\n"
            "Answer: X\n"
            "where X is one of A, B, C, or D.\n\n"
            "CRITICAL: Do NOT include any text, explanation, reasoning, or uncertainty after 'Answer: X'. Just the answer."
        )

        # 生成阶段：按实验设定，主图永远不传入 patches（Local Slices）
        response = generator.generate(
            global_image=image,
            patches=[],
            query=query_with_context,
        )
        torch.cuda.empty_cache()

        import re

        match = re.search(r"Answer:\s*([A-D])", response, re.IGNORECASE)
        if match:
            predicted = match.group(1).upper()
            is_correct = predicted == ground_truth
            print(f"预测：{predicted}，真实：{ground_truth}，匹配：{is_correct}")
            
            # 记录结果
            results.append({
                "sample_id": sample.get("id", f"sample_{i}"),
                "scenario": sample.get("scenario", "unknown"),
                "aspect": sample.get("aspect", "unknown"),
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": is_correct,
                "question": question,
            })
        else:
            print("✗ 无法解析答案")
            # 记录无法解析的情况
            results.append({
                "sample_id": sample.get("id", f"sample_{i}"),
                "scenario": sample.get("scenario", "unknown"),
                "aspect": sample.get("aspect", "unknown"),
                "predicted": None,
                "ground_truth": ground_truth,
                "correct": False,
                "question": question,
            })

    # 统计结果
    print("\n" + "=" * 60)
    print("基线测试结果统计（Biological 场景）")
    print("=" * 60)
    
    total_samples = len(results)
    valid_predictions = sum(1 for r in results if r["predicted"] is not None)
    correct_predictions = sum(1 for r in results if r.get("correct", False))
    
    print(f"总样本数：{total_samples}")
    print(f"有效预测数：{valid_predictions} (无法解析答案：{total_samples - valid_predictions})")
    print(f"正确数：{correct_predictions}")
    
    if valid_predictions > 0:
        accuracy = correct_predictions / valid_predictions
        print(f"准确率：{accuracy:.4f} ({accuracy * 100:.2f}%)")
    else:
        print("准确率：N/A (无有效预测)")
    
    # 按 aspect 统计
    aspect_stats: Dict[str, Dict[str, int]] = {}
    for r in results:
        aspect = r["aspect"]
        if aspect not in aspect_stats:
            aspect_stats[aspect] = {"total": 0, "correct": 0, "valid": 0}
        aspect_stats[aspect]["total"] += 1
        if r["predicted"] is not None:
            aspect_stats[aspect]["valid"] += 1
            if r.get("correct", False):
                aspect_stats[aspect]["correct"] += 1
    
    if aspect_stats:
        print("\n按 Aspect 统计：")
        print("-" * 60)
        print(f"{'Aspect':<30} {'准确率':<15} {'正确数/有效数/总数'}")
        print("-" * 60)
        for aspect, stats in sorted(aspect_stats.items()):
            if stats["valid"] > 0:
                acc = stats["correct"] / stats["valid"]
                print(
                    f"{aspect:<30} {acc:.4f} ({acc*100:.2f}%)  "
                    f"{stats['correct']}/{stats['valid']}/{stats['total']}"
                )
            else:
                print(
                    f"{aspect:<30} N/A (无有效预测)  "
                    f"{stats['correct']}/{stats['valid']}/{stats['total']}"
                )

    print("=" * 60)

    # 写出 MRAG-Bench 兼容的 jsonl 结果
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{run_name}.jsonl")
        import json

        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                record = {
                    "id": r["sample_id"],
                    "scenario": r["scenario"],
                    "aspect": r["aspect"],
                    "gt_choice": r["ground_truth"],
                    "output": (
                        f"Answer: {r['predicted']}"
                        if r["predicted"] is not None
                        else ""
                    ),
                    "prompt": r["question"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"✓ 已写入基线预测结果到 {out_path}")
    except Exception as e:
        print(f"⚠ 写入基线预测结果失败：{e}")

    print("\n✓ 基线测试完成")


def main() -> None:
    parser = argparse.ArgumentParser(description="MRAG-Bench Biological-Only Runner")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    offline_parser = subparsers.add_parser("offline", help="离线存储（仅 Biological 图片）")
    offline_parser.add_argument(
        "--image_dir",
        type=str,
        default=MRAG_IMAGE_CORPUS,
        help="图片目录",
    )
    offline_parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.7,
        help="SIMILAR_TO 相似度阈值",
    )
    offline_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="离线阶段限制图片数量（可选）",
    )
    offline_parser.add_argument(
        "--dataset",
        type=str,
        default="biological",
        help="数据集名称（用于 Milvus collection 隔离，默认：biological）",
    )
    offline_parser.add_argument(
        "--neo4j_uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j 连接 URI（默认：bolt://localhost:7687）",
    )
    offline_parser.add_argument(
        "--neo4j_database",
        type=str,
        default=None,
        help="Neo4j 数据库名称（用于多数据库隔离，如 'biological_db'。如果指定，将使用不同的数据库实例）",
    )
    offline_parser.add_argument(
        "--milvus_host",
        type=str,
        default="localhost",
        help="Milvus 服务器地址（默认：localhost）",
    )
    offline_parser.add_argument(
        "--milvus_port",
        type=str,
        default="19530",
        help="Milvus 服务器端口（默认：19530）",
    )
    offline_parser.add_argument(
        "--worker_rank",
        type=int,
        default=0,
        help="并行建库 worker rank（默认 0）",
    )
    offline_parser.add_argument(
        "--worker_world_size",
        type=int,
        default=1,
        help="并行建库 worker 总数（默认 1）",
    )
    offline_parser.add_argument(
        "--skip_relationships",
        action="store_true",
        help="跳过 SIMILAR_TO 构建（并行模式推荐）",
    )
    offline_parser.add_argument(
        "--reset_neo4j_dataset",
        action="store_true",
        help="删除 Neo4j 中当前 dataset 的 Image 节点后重建（谨慎使用）",
    )
    offline_parser.add_argument(
        "--reset_milvus_collection",
        action="store_true",
        help="删除当前 dataset 对应的 Milvus collection 后重建（谨慎使用）",
    )
    offline_parser.add_argument(
        "--milvus_flush_every",
        type=int,
        default=2000,
        help="Milvus 累计插入多少条后 flush 一次（0 表示仅最后 flush）",
    )
    offline_parser.add_argument(
        "--skip_embeddings",
        action="store_true",
        help="跳过 embedding 计算与写入（仅构建 Neo4j 概念图，速度更快）",
    )
    offline_parser.add_argument(
        "--skip_milvus_insert",
        action="store_true",
        help="不向 Milvus 写入向量（避免重复写入；通常与 --skip_embeddings 配合）",
    )

    parallel_parser = subparsers.add_parser("offline_parallel", help="多 GPU 并行离线建库")
    parallel_parser.add_argument("--image_dir", type=str, default=MRAG_IMAGE_CORPUS, help="图片目录")
    parallel_parser.add_argument("--similarity_threshold", type=float, default=0.7, help="SIMILAR_TO 阈值")
    parallel_parser.add_argument("--limit", type=int, default=0, help="限制图片数量（0 表示全部）")
    parallel_parser.add_argument("--dataset", type=str, default="biological", help="数据集名称")
    parallel_parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687", help="Neo4j URI")
    parallel_parser.add_argument("--neo4j_database", type=str, default=None, help="Neo4j database")
    parallel_parser.add_argument("--milvus_host", type=str, default="localhost", help="Milvus host")
    parallel_parser.add_argument("--milvus_port", type=str, default="19530", help="Milvus port")
    parallel_parser.add_argument("--gpus", type=str, default="4,5,6,7", help="GPU 列表，如 4,5,6,7")
    parallel_parser.add_argument("--reset_neo4j_dataset", action="store_true", help="先清理 Neo4j dataset")
    parallel_parser.add_argument("--reset_milvus_collection", action="store_true", help="先清理 Milvus collection")
    parallel_parser.add_argument("--milvus_flush_every", type=int, default=2000, help="flush 间隔（条数）")
    parallel_parser.add_argument("--skip_embeddings", action="store_true", help="跳过 embedding 计算与写入")
    parallel_parser.add_argument("--skip_milvus_insert", action="store_true", help="不向 Milvus 写入向量")

    rel_parser = subparsers.add_parser("build_relationships", help="单独构建 SIMILAR_TO 关系")
    rel_parser.add_argument("--dataset", type=str, default="biological", help="数据集名称")
    rel_parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687", help="Neo4j URI")
    rel_parser.add_argument("--neo4j_database", type=str, default=None, help="Neo4j database")
    rel_parser.add_argument("--milvus_host", type=str, default="localhost", help="Milvus host")
    rel_parser.add_argument("--milvus_port", type=str, default="19530", help="Milvus port")
    rel_parser.add_argument("--similarity_threshold", type=float, default=0.7, help="阈值")
    rel_parser.add_argument("--top_k", type=int, default=5, help="top_k")
    rel_parser.add_argument("--no_reset", action="store_true", help="不清理旧 SIMILAR_TO")

    online_parser = subparsers.add_parser("online", help="在线推理（仅 Biological 场景）")
    online_parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="在线阶段样本数量",
    )
    online_parser.add_argument(
        "--run_name",
        "--run-name",
        type=str,
        default="QWEN_RAG_FULL",
        help="运行名称（用于输出文件命名）",
    )
    online_parser.add_argument(
        "--dataset",
        type=str,
        default="biological",
        help="数据集名称（用于 Milvus collection 隔离，默认：biological）",
    )
    online_parser.add_argument(
        "--neo4j_uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j 连接 URI（默认：bolt://localhost:7687）",
    )
    online_parser.add_argument(
        "--neo4j_database",
        type=str,
        default=None,
        help="Neo4j 数据库名称（用于多数据库隔离，如 'biological_db'。如果指定，将使用不同的数据库实例）",
    )
    online_parser.add_argument(
        "--milvus_host",
        type=str,
        default="localhost",
        help="Milvus 服务器地址（默认：localhost）",
    )
    online_parser.add_argument(
        "--milvus_port",
        type=str,
        default="19530",
        help="Milvus 服务器端口（默认：19530）",
    )
    online_parser.add_argument(
        "--out_dir",
        type=str,
        default="results_bio_ablation/Qwen25VL7B",
        help="预测结果输出目录（jsonl）",
    )
    online_parser.add_argument(
        "--rag_mode",
        "--rag-mode",
        type=str,
        default="ours_full",
        choices=["ours_full", "ours", "qimg_only", "graph_only"],
        help="RAG 模式（ours_full / ours / qimg_only / graph_only）",
    )
    online_parser.add_argument(
        "--top_images",
        "--top-images",
        type=int,
        default=5,
        help="每个查询使用的检索图像上限",
    )
    # OURS fusion weights (for ablations)
    online_parser.add_argument("--alpha", type=float, default=1.0, help="OURS: 正实体 HAS 权重（默认 1.0）")
    online_parser.add_argument("--beta", type=float, default=2.0, help="OURS: 正实体 NOT_HAS 惩罚权重（默认 2.0）")
    online_parser.add_argument("--gamma", type=float, default=0.0, help="OURS: 负实体 HAS 惩罚权重（默认 0.0）")
    online_parser.add_argument("--delta", type=float, default=0.0, help="OURS: 负实体 NOT_HAS 奖励权重（默认 0.0）")
    online_parser.add_argument(
        "--lambda_conceptset",
        "--lambda-conceptset",
        type=float,
        default=1.0,
        help="OURS: 图检索得分权重（默认 1.0）",
    )
    online_parser.add_argument(
        "--lambda_qimg",
        "--lambda-qimg",
        type=float,
        default=1.0,
        help="OURS: 向量检索得分权重（默认 1.0）",
    )

    baseline_parser = subparsers.add_parser("baseline", help="基线测试（仅 Biological 场景）")
    baseline_parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="基线测试样本数量",
    )
    baseline_parser.add_argument(
        "--run_name",
        type=str,
        default="QWEN_NO_RAG",
        help="运行名称（用于输出文件命名）",
    )
    baseline_parser.add_argument(
        "--out_dir",
        type=str,
        default="results_bio_ablation/Qwen25VL7B",
        help="基线预测结果输出目录（jsonl）",
    )

    args = parser.parse_args()

    if args.mode == "offline":
        offline_store(
            image_dir=args.image_dir,
            similarity_threshold=args.similarity_threshold,
            limit=args.limit,
            dataset=args.dataset,
            neo4j_uri=getattr(args, "neo4j_uri", "bolt://localhost:7687"),
            neo4j_database=getattr(args, "neo4j_database", None),
            milvus_host=getattr(args, "milvus_host", "localhost"),
            milvus_port=getattr(args, "milvus_port", "19530"),
            worker_rank=getattr(args, "worker_rank", 0),
            worker_world_size=getattr(args, "worker_world_size", 1),
            skip_relationships=getattr(args, "skip_relationships", False),
            reset_neo4j_dataset=getattr(args, "reset_neo4j_dataset", False),
            reset_milvus_collection=getattr(args, "reset_milvus_collection", False),
            milvus_flush_every=getattr(args, "milvus_flush_every", 2000),
            skip_embeddings=getattr(args, "skip_embeddings", False),
            skip_milvus_insert=getattr(args, "skip_milvus_insert", False),
        )
    elif args.mode == "offline_parallel":
        offline_parallel(
            image_dir=args.image_dir,
            similarity_threshold=args.similarity_threshold,
            limit=args.limit,
            dataset=args.dataset,
            neo4j_uri=args.neo4j_uri,
            neo4j_database=args.neo4j_database,
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            gpus=args.gpus,
            reset_neo4j_dataset=args.reset_neo4j_dataset,
            reset_milvus_collection=args.reset_milvus_collection,
            milvus_flush_every=args.milvus_flush_every,
            skip_embeddings=getattr(args, "skip_embeddings", False),
            skip_milvus_insert=getattr(args, "skip_milvus_insert", False),
        )
    elif args.mode == "build_relationships":
        build_relationships(
            dataset=args.dataset,
            neo4j_uri=args.neo4j_uri,
            neo4j_database=args.neo4j_database,
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            similarity_threshold=args.similarity_threshold,
            top_k=args.top_k,
            reset_existing=not args.no_reset,
        )
    elif args.mode == "online":
        online_infer(
            limit=args.limit,
            dataset=args.dataset,
            neo4j_uri=getattr(args, "neo4j_uri", "bolt://localhost:7687"),
            neo4j_database=getattr(args, "neo4j_database", None),
            milvus_host=getattr(args, "milvus_host", "localhost"),
            milvus_port=getattr(args, "milvus_port", "19530"),
            run_name=getattr(args, "run_name", "QWEN_RAG_FULL"),
            out_dir=getattr(args, "out_dir", "results_bio_ablation/Qwen25VL7B"),
            rag_mode=getattr(args, "rag_mode", "ours_full"),
            top_images=getattr(args, "top_images", 5),
            alpha=getattr(args, "alpha", 1.0),
            beta=getattr(args, "beta", 2.0),
            gamma=getattr(args, "gamma", 0.0),
            delta=getattr(args, "delta", 0.0),
            lambda_conceptset=getattr(args, "lambda_conceptset", 1.0),
            lambda_qimg=getattr(args, "lambda_qimg", 1.0),
        )
    elif args.mode == "baseline":
        online_infer_baseline(
            limit=args.limit,
            run_name=getattr(args, "run_name", "QWEN_NO_RAG"),
            out_dir=getattr(args, "out_dir", "results_bio_ablation/Qwen25VL7B"),
        )


if __name__ == "__main__":
    main()
