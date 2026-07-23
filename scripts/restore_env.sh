#!/bin/bash
# restore_env.sh — 恢复默认模型环境（dev/main 产品分支）
# 模型权重/lens 保存在 .model_cache/，链接到 /tmp 即可（/tmp 链接重启后失效需重跑）
# 数据集下载（GraphRAG-Bench/BEIR）属实验环境，见 experiment 分支版本。

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE=$PROJECT/.model_cache

echo "=== Restoring model environment ==="

# 1. Link model weights to /tmp (avoid copying 15GB)
if [ ! -d /tmp/qwen25-7b-it-weights ]; then
    ln -s $CACHE/qwen25-7b-it-weights /tmp/qwen25-7b-it-weights
    echo "  Linked qwen25-7b-it-weights → /tmp"
else
    echo "  qwen25-7b-it-weights already in /tmp"
fi

# 2. Link lens to /tmp
if [ ! -d /tmp/jlens-qwen25-7b-it ]; then
    ln -s $CACHE/jlens-qwen25-7b-it /tmp/jlens-qwen25-7b-it
    echo "  Linked jlens-qwen25-7b-it → /tmp"
else
    echo "  jlens-qwen25-7b-it already in /tmp"
fi

# 3. Sanity check: weights dir complete (safetensors shards + index) and lens present
python3 - "$PROJECT" <<'EOF'
import sys
from pathlib import Path

def dir_complete(d: Path) -> bool:
    if not d.is_dir():
        return False
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        import json
        shards = {f for f in json.loads(idx.read_text())["weight_map"].values()}
        return all((d / s).exists() for s in shards)
    return any(d.glob("*.safetensors"))

proj = Path(sys.argv[1])
model = Path("/tmp/qwen25-7b-it-weights")
lens = Path("/tmp/jlens-qwen25-7b-it")
ok = dir_complete(model) and lens.exists()
print(f"  Model: {'OK' if dir_complete(model) else 'MISSING'}")
print(f"  Lens:  {'OK' if lens.exists() else 'MISSING'}")
if not ok:
    print("  ERROR: model environment incomplete — check .model_cache/")
    sys.exit(1)
print()
print("=== Environment ready ===")
EOF
