#!/usr/bin/env python3
"""
下载 Qwen2.5-VL-7B-Instruct 模型到 checkpoints 目录
使用 modelscope 国内镜像
"""

import os
from pathlib import Path


# 使用 modelscope 下载
def download_qwen_vl():
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    target_path = Path("./checkpoints/Qwen2.5-VL-7B-Instruct").resolve()

    print(f"🚀 正在从 ModelScope 下载: {model_id}")
    print(f"📂 目标路径: {target_path}")
    print(f"📊 模型大小: 约 15GB")

    try:
        from modelscope import snapshot_download

    except ImportError:
        print("\n❌ modelscope 未安装，正在安装...")
        os.system("pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple")
        from modelscope import snapshot_download

    try:
        snapshot_download(
            model_id,
            cache_dir=target_path.parent,
            revision="master",
        )

        print(f"\n✅ 下载成功！模型已保存到: {target_path}")

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("💡 建议检查：")
        print("   1. 网络连接是否正常")
        print("   2. 磁盘空间是否充足（需要约 20GB）")
        print("   3. conda 环境是否正确激活")


if __name__ == "__main__":
    download_qwen_vl()
