#!/usr/bin/env python3
"""Aggressive byte-range resumable downloader for flaky CDN connections.

Unlike curl -C - (which stalls on dropped connections and waits for timeout),
this downloader:
  - Opens with a Range header from the current file size
  - Reads in 1MB chunks, flushing to disk immediately
  - On ANY connection error, retries from the new file size within 2s
  - Verifies final size matches expected

Usage: python _range_dl.py URL OUT_PATH EXPECTED_SIZE
"""
import sys
import os
import time
import urllib.request
import urllib.error

CHUNK = 1024 * 1024  # 1MB


def get_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def download(url: str, out: str, expected: int, max_minutes: int = 120):
    start = time.time()
    deadline = start + max_minutes * 60
    attempt = 0
    while True:
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if have >= expected:
            print(f"\nCOMPLETE: {have}/{expected} bytes")
            return True
        if time.time() > deadline:
            print(f"\nTIMEOUT after {max_minutes}min at {have}/{expected}")
            return False
        attempt += 1
        headers = {"Range": f"bytes={have}-"}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(out, "ab") as f:
                last_print = have
                last_print_t = time.time()
                while True:
                    chunk = r.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                    have += len(chunk)
                    now = time.time()
                    if now - last_print_t > 10:
                        rate = (have - last_print) / (now - last_print_t) / 1e6
                        pct = have * 100 / expected
                        eta = (expected - have) / (rate * 1e6) if rate > 0 else 0
                        print(f"  {have/1e6:.0f}/{expected/1e6:.0f}MB ({pct:.0f}%) "
                              f"{rate:.1f}MB/s ETA {eta/60:.0f}min [attempt {attempt}]",
                              flush=True)
                        last_print = have
                        last_print_t = now
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            print(f"  retry at {have/1e6:.0f}MB: {type(e).__name__}", flush=True)
            time.sleep(2)
            continue
        # loop back to check size


if __name__ == "__main__":
    url, out, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ok = download(url, out, expected)
    sys.exit(0 if ok else 1)
