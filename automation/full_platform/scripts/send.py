import argparse
import concurrent.futures
import json
from pathlib import Path
import time

import requests


def load_images(image_dir: Path):
    images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"No images found in {image_dir}")
    return images


def send_request(url: str, image_path: Path, timeout: float):
    with image_path.open("rb") as fh:
        files = {"file": (image_path.name, fh, "image/jpeg")}
        start = time.time()
        resp = requests.post(url, files=files, timeout=timeout)
        latency = time.time() - start
        return resp.status_code, latency, resp.text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--images", required=True, help="Directory with test images")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--out", default=None, help="Optional JSON output file")
    args = parser.parse_args()

    images = load_images(Path(args.images))
    results = []
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for i in range(args.requests):
            image_path = images[i % len(images)]
            futures.append(pool.submit(send_request, args.url, image_path, args.timeout))

        for future in concurrent.futures.as_completed(futures):
            status, latency, body = future.result()
            results.append({"status": status, "latency": latency, "body": body})

    duration = time.time() - start
    latencies = [r["latency"] for r in results]
    summary = {
        "requests": len(results),
        "duration": duration,
        "rps": len(results) / max(duration, 1e-6),
        "latency_avg": sum(latencies) / max(len(latencies), 1),
        "latency_p95": sorted(latencies)[int(0.95 * len(latencies)) - 1],
    }

    print(json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=2))


if __name__ == "__main__":
    main()
