"""轻量 OpenAI 兼容 LLM 服务——用 transformers 直接加载 Qwen2.5-7B 4-bit。

为 SOTA 框架（Fast-GraphRAG / HippoRAG2 等）提供 OpenAI 兼容的 /v1/chat/completions
和 /v1/embeddings 端点，不依赖 vLLM/Ollama。

所有方法用同一个 Qwen2.5-7B-Instruct（4-bit NF4）+ bge-m3，确保公平对比。

Usage:
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase26_local_llm_server --port 8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import detect_model, load_model, _model_dir_complete
from experiments.embed_cache import CachedBgeM3Provider


# Global model handles (loaded once at startup)
_model = None
_tokenizer = None
_embed = None


def load_models():
    global _model, _tokenizer, _embed
    print("[startup] Loading Qwen2.5-7B-Instruct (4-bit)...", flush=True)
    cand = detect_model()
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    _model, _tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])
    _model.eval()
    print(f"[startup] Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    print("[startup] Loading bge-m3...", flush=True)
    _embed = CachedBgeM3Provider()
    print("[startup] Embedding model loaded.", flush=True)


def generate_completion(messages: list[dict], max_tokens: int = 512,
                        temperature: float = 0.0) -> str:
    """Generate text using the local Qwen model."""
    # Apply chat template
    if _tokenizer and hasattr(_tokenizer, "apply_chat_template"):
        try:
            prompt = _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    input_ids = _tokenizer.encode(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        output = _model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_ids.shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using bge-m3."""
    vecs = _embed.embed(texts)
    return [v.tolist() for v in vecs]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[api] {args[0]}", flush=True)

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode()

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, 400)
            return

        if self.path == "/v1/chat/completions":
            self._handle_chat(data)
        elif self.path == "/v1/embeddings":
            self._handle_embed(data)
        else:
            self._send_json({"error": f"unknown path: {self.path}"}, 404)

    def do_GET(self):
        if self.path == "/v1/models" or self.path == "/v1/model":
            self._send_json({
                "object": "list",
                "data": [{"id": "qwen2.5-7b-instruct", "object": "model"}]
            })
        elif self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": f"unknown path: {self.path}"}, 404)

    def _handle_chat(self, data):
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 512)
        temperature = data.get("temperature", 0.0)
        model = data.get("model", "qwen2.5-7b-instruct")

        t0 = time.perf_counter()
        content = generate_completion(messages, max_tokens, temperature)
        elapsed = time.perf_counter() - t0

        # OpenAI-compatible response format
        self._send_json({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        })

    def _handle_embed(self, data):
        input_texts = data.get("input", [])
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        model = data.get("model", "bge-m3")

        t0 = time.perf_counter()
        embeddings = embed_texts(input_texts)
        elapsed = time.perf_counter() - t0

        self._send_json({
            "object": "list",
            "model": model,
            "data": [
                {"object": "embedding", "index": i, "embedding": emb}
                for i, emb in enumerate(embeddings)
            ],
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    load_models()

    server = HTTPServer((args.host, args.port), Handler)
    print(f"\n[server] Listening on http://{args.host}:{args.port}", flush=True)
    print(f"[server] Endpoints:", flush=True)
    print(f"  POST /v1/chat/completions  (Qwen2.5-7B-Instruct 4-bit)", flush=True)
    print(f"  POST /v1/embeddings        (bge-m3)", flush=True)
    print(f"  GET  /v1/models", flush=True)
    print(f"  GET  /health", flush=True)
    print(f"\n[server] Ready for SOTA framework connections.", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
