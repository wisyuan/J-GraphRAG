"""Import-safety: importing the package must not pull heavy ML deps."""
from __future__ import annotations

import subprocess
import sys

HEAVY = ("torch", "transformers", "FlagEmbedding", "jlens")

_SNIPPET = """
import sys
import jgraphrag
import jgraphrag.extract
import jgraphrag.index
import jgraphrag.pipeline
import jgraphrag.retrieve
import jgraphrag.providers.base
import jgraphrag.providers.bge_m3
import jgraphrag.providers.qwen_jlens
import jgraphrag.stores.local
heavy = {heavy!r}
bad = sorted(m for m in sys.modules if m.split(".")[0] in heavy)
if bad:
    print("LEAKED:", bad)
    sys.exit(1)
print("clean")
"""


def test_import_jgraphrag_has_no_heavy_side_effects():
    proc = subprocess.run(
        [sys.executable, "-c", _SNIPPET.format(heavy=HEAVY)],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f"heavy modules leaked into sys.modules: {proc.stdout} {proc.stderr}")
    assert "clean" in proc.stdout
