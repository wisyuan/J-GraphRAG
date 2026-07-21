#!/bin/bash
# restore_env.sh — 一键恢复模型环境
# 模型权重/lens 已保存在 .model_cache/，只需恢复到 /tmp + 下载数据集

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

# 3. Download datasets if not cached
source $PROJECT/.venv/bin/activate 2>/dev/null

python3 -c "
import os, sys
os.environ['HF_HUB_DISABLE_XET'] = '1'
sys.path.insert(0, '$PROJECT/crates/lincle/python')

# Check model
from experiments.phase10_jlens_stage1 import detect_model, _model_dir_complete
c = detect_model()
ok = _model_dir_complete(c['local_model_dir']) and __import__('pathlib').Path(c['local_lens_path']).exists()
print(f'  Model: {\"OK\" if ok else \"MISSING\"} (complete={_model_dir_complete(c[\"local_model_dir\"])})')

if not ok:
    print('  ERROR: Model weights missing!')
    sys.exit(1)

# Download BEIR datasets if needed
from beir.util import download_and_unzip
from beir.datasets.data_loader import GenericDataLoader
for ds in ['nfcorpus', 'scifact']:
    p = f'/tmp/beir-datasets/{ds}'
    if not os.path.exists(p):
        print(f'  Downloading {ds}...')
        download_and_unzip(f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds}.zip', '/tmp/beir-datasets')
    corpus, _, _ = GenericDataLoader(data_folder=p).load()
    print(f'  {ds}: {len(corpus)} docs')

# Download GraphRAG-Bench if needed
from huggingface_hub import hf_hub_download
import subprocess
for f in ['Datasets/Corpus/novel.json', 'Datasets/Corpus/medical.json',
          'Datasets/Questions/medical_questions.json', 'Datasets/Questions/novel_questions.json']:
    try:
        hf_hub_download('GraphRAG-Bench/GraphRAG-Bench', f, repo_type='dataset', local_dir='/tmp/graphrag-bench')
    except: pass
subprocess.run('cd /tmp/graphrag-bench && ln -sf Datasets/Corpus/medical.json medical.json && ln -sf Datasets/Corpus/novel.json novel.json && ln -sf Datasets/Questions/medical_questions.json medical_questions.json && ln -sf Datasets/Questions/novel_questions.json novel_questions.json', shell=True)
print('  GraphRAG-Bench: OK')

print()
print('=== Environment ready ===')
"
