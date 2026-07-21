"""统一语料加载器——把 4 个领域的原始数据转为 FineRecord 粒度。

4 个领域：
  - pi（TypeScript 代码）→ tree-sitter 替代（regex 符号切分）
  - enterprise（业务文档/SQL/CSV）→ 文件/表级
  - NFCorpus（医学营养文本，BEIR）→ 段落切分
  - SciFact（科学论文摘要，BEIR）→ 摘要整篇（已是 FineRecord 粒度）

每个加载器返回 list[tuple[str, str]]：(fine_record_id, fine_record_text)。
"""
from __future__ import annotations

from pathlib import Path


def load_pi_fine_records(repo_path: str, max_files: int = 50) -> list[tuple[str, str]]:
    """pi TypeScript → 符号级 FineRecord（复用 split_file_to_symbols）。"""
    from experiments.fine_record_dispersion import split_file_to_symbols

    repo = Path(repo_path)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))[:max_files]
    records = []
    for f in files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        for sym in split_file_to_symbols(content):
            rid = f"{f.relative_to(repo)}::{sym['name']}"
            text = f"{sym['kind']} {sym['name']}:\n{sym['body']}"
            records.append((rid, text))
    return records


def load_enterprise_fine_records(scenario_path: str | None = None) -> list[tuple[str, str]]:
    """m5 enterprise → 文档/脚本/CSV/DB 表级 FineRecord。
    复用 mixed_corpus_density.load_enterprise 的逻辑。"""
    from experiments.mixed_corpus_density import load_enterprise

    if scenario_path is None:
        repo = Path(__file__).resolve().parents[1]
        scenario_path = str(repo / "experiments" / "m5" / "enterprise_scenario")

    raw = load_enterprise(scenario_path)  # list[(id, text)]
    return [(rid, text) for rid, text in raw]


def _split_into_paragraphs(text: str, max_chars: int = 500) -> list[str]:
    """把自然语言文本切成段落级 FineRecord（~500 字符，和代码符号 body 一致）。"""
    import re
    # 按句号/换行分割，合并到 ~max_chars
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    paragraphs = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 > max_chars and current:
            paragraphs.append(current)
            current = sent
        else:
            current = f"{current} {sent}".strip() if current else sent
    if current:
        paragraphs.append(current)
    return paragraphs if paragraphs else [text[:max_chars]]


def load_beir_fine_records(dataset_name: str,
                           data_path: str = "/tmp/beir-datasets",
                           max_docs: int = 1000) -> list[tuple[str, str]]:
    """BEIR 数据集 → 段落级 FineRecord。

    NFCorpus/SciFact 的文档是摘要/短文，切为 ~500 字符段落。
    max_docs 限制文档数（控计算量；Phase 3 图质量评估不需要全量）。
    """
    from beir.datasets.data_loader import GenericDataLoader

    folder = Path(data_path) / dataset_name
    corpus, queries, qrels = GenericDataLoader(data_folder=str(folder)).load()

    records = []
    doc_ids = list(corpus.keys())[:max_docs]
    for doc_id in doc_ids:
        text = corpus[doc_id].get("text", "") or corpus[doc_id].get("title", "")
        if not text or len(text.strip()) < 20:
            continue
        paragraphs = _split_into_paragraphs(text)
        for pi, para in enumerate(paragraphs):
            rid = f"{dataset_name}:{doc_id}:p{pi}"
            records.append((rid, para))
    return records


def load_all_corpora(repo_path: str = "/tmp/pi-repo",
                     max_pi_files: int = 50,
                     max_beir_docs: int = 1000) -> dict[str, list[tuple[str, str]]]:
    """加载全部 4 个领域语料，返回 {domain_name: [(id, text), ...]}。"""
    corpora = {}

    print("  loading pi...", end="", flush=True)
    corpora["pi"] = load_pi_fine_records(repo_path, max_pi_files)
    print(f" {len(corpora['pi'])} fine records")

    print("  loading enterprise...", end="", flush=True)
    corpora["enterprise"] = load_enterprise_fine_records()
    print(f" {len(corpora['enterprise'])} fine records")

    for dataset in ["nfcorpus", "scifact"]:
        print(f"  loading {dataset}...", end="", flush=True)
        corpora[dataset] = load_beir_fine_records(dataset, max_docs=max_beir_docs)
        print(f" {len(corpora[dataset])} fine records")

    return corpora


if __name__ == "__main__":  # pragma: no cover
    corpora = load_all_corpora()
    for name, records in corpora.items():
        print(f"\n{name}: {len(records)} records")
        print(f"  sample: {records[0][0]} → {records[0][1][:100]}...")
