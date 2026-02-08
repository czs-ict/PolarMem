# PolarMem

## 项目结构

```
PolarMem/
├── src/
│   ├── backbone/         
│   ├── database/        
│   ├── logic/            
│   └── pipeline/          
├── scripts/
│   └── benchmark_biological.py  
├── utils/                  
├── download_qwen_vl.py     
├── score.py                
└── README.md
```

## 数据准备

1. **MRAG-Bench 数据**  
   - 将测试集 parquet 放在 `MRAG_DATA_DIR`（默认 `data/mrag_bench_raw/data`），脚本会读取 `test-*.parquet`。  
   - 将图像库放在 `MRAG_IMAGE_CORPUS`（默认 `data/mrag_bench_image_corpus`），文件名为 `{image_id}.jpg`（或 `.png`/`.jpeg`）。

2. **Qwen2.5-VL 模型**  
   - 使用项目内脚本下载（国内可用 ModelScope）：
     ```bash
     python download_qwen_vl.py
     ```
   - 或将已有 checkpoint 路径设为 `QWEN_MODEL_PATH`。

3. **Neo4j 与 Milvus**  
   - 本地或远程启动 Neo4j、Milvus，记下 URI/主机/端口，在线与离线脚本均支持通过参数指定。

## 使用方式

所有 Biological 相关流程由 `scripts/benchmark_biological.py` 提供子命令。

### 离线建库（单机）

```bash
cd /path/to/PolarMem
python scripts/benchmark_biological.py offline \
  --image_dir "$MRAG_IMAGE_CORPUS" \
  --dataset biological \
  --neo4j_uri bolt://localhost:7687 \
  --milvus_host localhost \
  --milvus_port 19530
```

可选：`--limit N` 限制处理图片数；`--reset_neo4j_dataset` / `--reset_milvus_collection` 清空后重建；`--skip_relationships` 跳过 SIMILAR_TO 构建等。

### 离线建库（多 GPU 并行）

```bash
python scripts/benchmark_biological.py offline_parallel \
  --image_dir "$MRAG_IMAGE_CORPUS" \
  --dataset biological \
  --gpus 0,1,2,3
```

### 单独构建 SIMILAR_TO 关系

若离线阶段使用了 `--skip_relationships`，可后续单独建关系：

```bash
python scripts/benchmark_biological.py build_relationships \
  --dataset biological \
  --similarity_threshold 0.7
```

### 在线推理（RAG）

```bash
python scripts/benchmark_biological.py online \
  --limit 100 \
  --run_name QWEN_RAG_FULL \
  --rag_mode ours_full \
  --top_images 5 \
  --out_dir results_bio_ablation/Qwen25VL7B
```

`--rag_mode` 可选：`ours_full`、`ours`、`qimg_only`、`graph_only`；融合权重可用 `--alpha`、`--beta`、`--lambda_conceptset`、`--lambda_qimg` 等调节。

### 基线（无 RAG）

```bash
python scripts/benchmark_biological.py baseline \
  --limit 100 \
  --run_name QWEN_NO_RAG \
  --out_dir results_bio_ablation/Qwen25VL7B
```

### 打分

模型输出为 json/jsonl 时，可使用 `score.py` 计算整体与分场景准确率：

```bash
python score.py --input_file results_bio_ablation/Qwen25VL7B/QWEN_RAG_FULL.jsonl
```

结果写入 `results/` 目录。

