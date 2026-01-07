"""Client-side smoke tests for the embedding server.

Usage:
Run:
      uv run ./tests/embedding_server_py/client_test.py --server_url http://127.0.0.1:3752
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import numpy

from PIL import Image

MODEL_NAMES = ["siglip2", "dinov2"]


TEXTS = [
    "A photo of a cat lying on a red couch",
    "A red car parked on a street",
    "hello world",
]


def _urljoin(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _http_get_json(url: str, timeout: float = 10) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
        return json.loads(payload)


def mark_in_out(f):
    def wrapper(*args, **kwargs):
        print(f"[->Entering] {f.__name__}")
        result = f(*args, **kwargs)
        print(f"[<-Exit-ing] {f.__name__}")
        return result
    return wrapper


@dataclass
class HttpResult:
    status: int
    body: str


def _http_post_json(url: str, data: dict[str, Any], timeout: float = 60) -> HttpResult:
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return HttpResult(status=getattr(resp, "status", 200), body=body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else str(e)
        return HttpResult(status=e.code, body=body)


def _make_base64_png_data_uri(size: int = 64) -> str:
    img = Image.new("RGB", (size, size), (200, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _write_png_to_file(path: str, size: int = 64) -> None:
    img = Image.new("RGB", (size, size), (50, 120, 200))
    img.save(path, format="PNG")


def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def _assert_embedding(vec: list[float], min_dim: int = 32) -> None:
    assert isinstance(vec, list)
    assert len(vec) >= min_dim
    n = _l2_norm(vec)
    # L2-normalized embeddings should be close to 1.
    assert 0.98 <= n <= 1.02, f"unexpected norm={n}"


def _dot_similarity(a: list[float], b: list[float]) -> float:
    """Dot product similarity for L2-normalized embeddings."""
    return float(numpy.dot(numpy.asarray(a, dtype=numpy.float32), numpy.asarray(b, dtype=numpy.float32)))


def _make_mixed_img_list() -> tuple[list[str], Callable[[], None]]:
    tmp_dir = tempfile.mkdtemp(prefix="embedding-imgs-")
    file_path = os.path.join(tmp_dir, "local.png")
    _write_png_to_file(file_path)

    http_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    base64_img = _make_base64_png_data_uri()
    file_uri = Path(file_path).resolve().as_uri()

    imgs = [base64_img, http_url, file_uri, base64_img]
    print(imgs)

    def cleanup() -> None:
        try:
            pass
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return imgs, cleanup


@mark_in_out
def test_text_image_retrieval_sanity_siglip2(server_url: str) -> None:
    """Sanity-check that SigLIP2 text/image embeddings are in the same space.

    This is intentionally lightweight: a cat-related query should score higher
    against a COCO cats image than an unrelated car query.
    """
    coco_cats_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img_emb = test_image_single(server_url, "siglip2", coco_cats_url)

    # Query text candidates (keep deterministic, no prompt templates).
    cat_query = "two cats lying on a couch"
    car_query = "a red car parked on a street"
    animal_query = "some animals in the image"

    cat_emb = test_text_single(server_url, "siglip2", cat_query)
    car_emb = test_text_single(server_url, "siglip2", car_query)
    animal_emb = test_text_single(server_url, "siglip2", animal_query)

    sim_cat = _dot_similarity(cat_emb, img_emb)
    sim_car = _dot_similarity(car_emb, img_emb)
    sim_animal = _dot_similarity(animal_emb, img_emb)
    print(f"Sanity text↔image similarity: cat={sim_cat:.4f} animal={sim_animal:.4f} car={sim_car:.4f}")

    # Allow a small margin; we mainly want to catch pathological near-zero/garbage text embeddings.
    assert sim_cat > sim_car - 0.01, f"unexpected ranking: cat={sim_cat} car={sim_car}"

def post_embeddings(server_url: str, model: str, inputs: list[str]) -> tuple[int, dict[str, Any] | None, str]:
    url = _urljoin(server_url, "/v1/embeddings")
    res = _http_post_json(url, {"input": inputs, "model": model, "encoding_format": "float"})
    try:
        parsed = json.loads(res.body)
    except Exception:
        parsed = None
    return res.status, parsed, res.body

@mark_in_out
def test_image_single(server_url: str, model_name: str, img_input: str) -> list[float]:
    status, parsed, raw = post_embeddings(server_url, model_name, [img_input])
    assert status == 200, f"expected 200 got {status}: {raw}"
    assert parsed is not None, f"invalid JSON response: {raw}"
    assert parsed.get("model") == model_name
    emb = parsed["data"][0]["embedding"]
    _assert_embedding(emb)
    print(f"Embedding stats - Mean: {numpy.mean(emb):.6f}, Std: {numpy.std(emb):.6f}")
    # print(emb)
    return emb

@mark_in_out
def test_image_batch(server_url: str, model_name: str, img_inputs: list[str]) -> list[list[float]]:
    status, parsed, raw = post_embeddings(server_url, model_name, img_inputs)
    assert status == 200, f"expected 200 got {status}: {raw}"
    assert parsed is not None, f"invalid JSON response: {raw}"
    assert len(parsed.get("data", [])) == len(img_inputs)
    out: list[list[float]] = []
    for item in parsed["data"]:
        _assert_embedding(item["embedding"])
        print(f"Embedding stats - Mean: {numpy.mean(item['embedding']):.6f}, Std: {numpy.std(item['embedding']):.6f}")
        out.append(item["embedding"])
    return out

@mark_in_out
def test_text_single(server_url: str, model_name: str, text: str) -> list[float]:
    status, parsed, raw = post_embeddings(server_url, model_name, [text])
    assert status == 200, f"expected 200 got {status}: {raw}"
    assert parsed is not None, f"invalid JSON response: {raw}"
    emb = parsed["data"][0]["embedding"]
    _assert_embedding(emb)
    print(f"Embedding stats - Mean: {numpy.mean(emb):.6f}, Std: {numpy.std(emb):.6f}")
    return emb

@mark_in_out
def test_text_batch(server_url: str, model_name: str, texts: list[str]) -> list[list[float]]:
    status, parsed, raw = post_embeddings(server_url, model_name, texts)
    assert status == 200, f"expected 200 got {status}: {raw}"
    assert parsed is not None, f"invalid JSON response: {raw}"
    out: list[list[float]] = []
    for item in parsed["data"]:
        _assert_embedding(item["embedding"])
        print(f"Embedding stats - Mean: {numpy.mean(item['embedding']):.6f}, Std: {numpy.std(item['embedding']):.6f}")
        out.append(item["embedding"])
    return out

@mark_in_out
def test_mixed_inputs_siglip2(server_url: str) -> None:
    img = _make_base64_png_data_uri()
    mixed = [TEXTS[0], img, TEXTS[1], img]
    status, parsed, raw = post_embeddings(server_url, "siglip2", mixed)
    assert status == 200, f"expected 200 got {status}: {raw}"
    assert parsed is not None, f"invalid JSON response: {raw}"
    assert [d["index"] for d in parsed["data"]] == list(range(len(mixed)))
    for d in parsed["data"]:
        _assert_embedding(d["embedding"])
        print(f"Embedding stats - Mean: {numpy.mean(d['embedding']):.6f}, Std: {numpy.std(d['embedding']):.6f}")

@mark_in_out
def test_dinov2_rejects_text(server_url: str) -> None:
    status, parsed, raw = post_embeddings(server_url, "dinov2", ["this is text"])
    assert status == 400, f"expected 400 got {status}: {raw}"

@mark_in_out
def start_server_as_child_process(server_url: str) -> subprocess.Popen:
    """Starts the embedding server as a child process and waits for it to be ready."""
    server_script = os.path.join("src", "embedding_server_py", "main_es.py")
    if not os.path.exists(server_script):
        # Try relative to this script if not in root
        server_script = os.path.join(os.path.dirname(__file__), "..", "..", "src", "embedding_server_py", "main_es.py")
    
    print(f"Starting server: {server_script}")
    env = os.environ.copy()
    env.setdefault("EMBEDDING_SERVER_ALLOW_LOCAL_FILE_URIS", "1")
    
    # Use sys.executable to ensure we use the same environment
    process = subprocess.Popen(
        [sys.executable, server_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    
    # Start a thread to continuously read and print server output
    def print_server_output():
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"[SERVER] {line.rstrip()}")
    
    output_thread = threading.Thread(target=print_server_output, daemon=True)
    output_thread.start()
    
    # Wait for server to start by polling /health
    health_url = _urljoin(server_url, "/health")
    max_retries = 60
    print(f"Waiting for server at {health_url}...")
    for i in range(max_retries):
        if process.poll() is not None:
            # Process died - wait a moment for output thread to catch up
            time.sleep(0.5)
            raise RuntimeError(f"Server process exited prematurely with code {process.returncode}. Check output above.")
        
        try:
            with urllib.request.urlopen(health_url, timeout=1) as resp:
                if resp.status == 200:
                    print("Server is up and healthy!")
                    return process
        except Exception:
            time.sleep(1)
            if i % 5 == 0 and i > 0:
                print(f"Still waiting... ({i}/{max_retries})")
    
    process.terminate()
    raise RuntimeError("Server failed to start and become healthy in time")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", default="http://127.0.0.1:3752")
    parser.add_argument("--start_server", action='store_false', help="Start the server as a child process")
    args = parser.parse_args()

    print(args._get_kwargs())

    server_url: str = args.server_url
    print("TESTING", server_url)

    server_proc = None
    if args.start_server:
        server_proc = start_server_as_child_process(server_url)

    try:
        # Basic endpoints
        health = _http_get_json(_urljoin(server_url, "/health"), timeout=10)
        assert health.get("status") == "ok", health

        models = _http_get_json(_urljoin(server_url, "/v1/models"), timeout=10)
        model_ids = {m["id"] for m in models.get("data", [])}
        for m in MODEL_NAMES:
            assert m in model_ids, models

        # Image tests
        img = _make_base64_png_data_uri()
        mixed_img_list, mixed_img_cleanup = _make_mixed_img_list()
        try:
            test_image_single(server_url, "siglip2", img)
            test_image_single(server_url, "dinov2", img)
            test_image_batch(server_url, "siglip2", mixed_img_list)
            test_image_batch(server_url, "dinov2", mixed_img_list)
        finally:
            mixed_img_cleanup()

        # Text tests (SigLIP2 supports text)
        test_text_single(server_url, "siglip2", TEXTS[0])
        test_text_batch(server_url, "siglip2", TEXTS)

        # Text↔image retrieval sanity (SigLIP2)
        test_text_image_retrieval_sanity_siglip2(server_url)

        # Mixed input ordering (SigLIP2)
        test_mixed_inputs_siglip2(server_url)

        # Expected error behavior
        test_dinov2_rejects_text(server_url)

        print("ALL TESTS PASSED")
        return 0
    finally:
        if server_proc:
            print("Stopping server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()
            print("Server stopped.")


if __name__ == "__main__":
    raise SystemExit(main())
    

