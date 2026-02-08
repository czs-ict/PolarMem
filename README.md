# PolarMem

## 项目结构

```
PolarMem/
├── src/
│   ├── backbone/          # 视觉骨干（Qwen2.5-VL 封装）
│   ├── database/          # Neo4j、Milvus 客户端
│   ├── logic/             # 概念提取、Otsu 阈值、切片等
│   └── pipeline/          # 检索器、生成器
├── scripts/
│   └── benchmark_biological.py   # Biological 场景离线建库 + 在线/基线评测
├── utils/                  # 数据加载、答案抽取（含 GPT 抽取）
├── download_qwen_vl.py     # 下载 Qwen2.5-VL 模型（ModelScope）
├── score.py                # 对模型输出进行打分（准确率等）
└── README.md
```

## 环境与依赖

- Python 3.9+
- PyTorch、CUDA（GPU 推理）
- 主要依赖：`transformers`、`qwen-vl-utils`、`neo4j`、`pymilvus`、`PIL`、`pyarrow`、`tqdm`、`scikit-image`
- 可选（答案抽取）：`openai`（需配置 `OPENAI_API_KEY`）

建议使用 conda/venv 创建虚拟环境后安装：

```bash
pip install torch torchvision
pip install transformers qwen-vl-utils
pip install neo4j pymilvus
pip install Pillow pyarrow tqdm scikit-image
# 若使用 score.py 的 GPT 抽取
pip install openai
```

## 环境变量（开源友好配置）

以下配置均通过环境变量或命令行参数指定，**请勿在代码中写入真实密钥或本机路径**。

| 变量名 | 说明 | 默认/示例 |
|--------|------|-----------|
| `QWEN_MODEL_PATH` | Qwen2.5-VL 模型路径或 HuggingFace 模型 ID | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `MRAG_DATA_DIR` | MRAG-Bench 测试集 parquet 所在目录 | `data/mrag_bench_raw/data` |
| `MRAG_IMAGE_CORPUS` | MRAG 图像库目录（按 image_id 存图） | `data/mrag_bench_image_corpus` |
| `IMAGE_ROOT` | 在线推理时检索到的图像所在根目录 | `.` |
| `OPENAI_API_KEY` | OpenAI API Key（仅 score.py 中 GPT 抽取需要） | 无默认，需自行设置 |
| `OPENAI_ORG` | OpenAI 组织 ID（可选） | 无默认 |

Neo4j / Milvus 的连接与认证通过脚本参数传入（如 `--neo4j_uri`、`--milvus_host`），生产环境建议从环境变量读取后再传参。

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

结果写入 `results/` 目录。若部分样本需 GPT 抽取答案，请先设置 `OPENAI_API_KEY`。

