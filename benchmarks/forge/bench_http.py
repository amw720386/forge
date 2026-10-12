"""Matched loopback SSE serving benchmark, standard library only.

Counts generated token events, including empty decoded pieces; fails on HTTP or
SSE errors. Native llama /completion provides one event per generated token.
EOS is enabled for both servers; actual outputs, not requested limits, are counted.
"""

import argparse
import concurrent.futures
import json
import math
import statistics
import subprocess
import threading
import time
import urllib.request

PROMPTS = {
    "short": "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
    "medium": "<|im_start|>user\n"
    + "Explain how a computer stores and retrieves information. " * 20
    + "<|im_end|>\n<|im_start|>assistant\n",
    "long": "<|im_start|>user\n"
    + "Explain how a computer stores and retrieves information. " * 100
    + "<|im_end|>\n<|im_start|>assistant\n",
}


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.load(r)


def request(args, prompt, barrier, delay_ms=0):
    payload = {"prompt": prompt, "stream": True, "temperature": 0}
    endpoint = "/v1/completions" if args.runtime == "forge" else "/completion"
    if args.runtime == "forge":
        payload["max_tokens"] = args.max_new
    else:
        payload.update(n_predict=args.max_new, cache_prompt=False, seed=0)
    req = urllib.request.Request(
        args.url + endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    if delay_ms:
        time.sleep(delay_ms / 1000)
    start = time.perf_counter()
    first = None
    tokens = 0
    timings = None
    text = ""
    with urllib.request.urlopen(req, timeout=120) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if "error" in event:
                raise RuntimeError(event["error"])
            if args.runtime == "forge":
                piece = event["choices"][0]["text"]
                tokens += 1
            else:
                timings = event.get("timings", timings)
                if event.get("stop", False):
                    continue
                piece = event.get("content", "")
                tokens += 1
            text += piece
            if first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    return dict(
        start=start,
        end=end,
        latency_s=end - start,
        ttft_s=None if first is None else first - start,
        tokens=tokens,
        text=text,
        server_timings=timings,
    )


def quantile(values, q):
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def run(args, concurrency, workload):
    barrier = threading.Barrier(concurrency)
    prompts = [
        PROMPTS[workload]
        if workload != "mixed"
        else PROMPTS[["short", "medium", "long"][i % 3]]
        for i in range(concurrency)
    ]
    before = get(args.url + "/metrics") if args.runtime == "forge" else None
    observed = []
    kv_observed = []
    scheduled_observed = []
    stop = threading.Event()
    vram = []

    def monitor_vram():
        while not stop.is_set():
            try:
                sample = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=True,
                )
                vram.append(float(sample.stdout.splitlines()[0]))
            except Exception:
                pass
            stop.wait(0.1)

    def observe():
        while not stop.is_set():
            try:
                data = get(
                    args.url + ("/metrics" if args.runtime == "forge" else "/slots")
                )
                active = (
                    data.get("active_decode", 0) + data.get("active_prefill", 0)
                    if isinstance(data, dict)
                    else sum(bool(s.get("is_processing")) for s in data)
                )
                observed.append(active)
                if isinstance(data, dict) and args.runtime == "forge":
                    capacity = data.get("kv_num_blocks", 0)
                    allocated = data.get("kv_allocated_blocks", 0)
                    kv_observed.append((allocated, capacity))
                    scheduled_observed.append(
                        data.get("scheduled_decode_tokens", 0)
                        + data.get("scheduled_prefill_tokens", 0)
                    )
            except Exception:
                pass
            stop.wait(0.005)

    observer = threading.Thread(target=observe)
    memory_observer = threading.Thread(target=monitor_vram)
    observer.start()
    memory_observer.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(
                pool.map(
                    lambda pair: request(
                        args, pair[1], barrier, pair[0] * args.arrival_gap_ms
                    ),
                    enumerate(prompts),
                )
            )
    finally:
        stop.set()
        observer.join()
        memory_observer.join()
    wall = max(r["end"] for r in results) - min(r["start"] for r in results)
    for r in results:
        r.pop("start")
        r.pop("end")
    after = get(args.url + "/metrics") if args.runtime == "forge" else None
    stage_delta = {}
    if before is not None:
        for name, value in after.get("stages", {}).items():
            old = before.get("stages", {}).get(name, {})
            count = value["count"] - old.get("count", 0)
            ns = value["total_ns"] - old.get("total_ns", 0)
            stage_delta[name] = dict(
                count=count, total_ns=ns, mean_ms=ns / count / 1e6 if count else None
            )
    return dict(
        server_stage_deltas=stage_delta,
        observed_active_mean=statistics.mean(observed) if observed else None,
        concurrency=concurrency,
        workload=workload,
        wall_s=wall,
        aggregate_tok_s=sum(r["tokens"] for r in results) / wall,
        observed_kv_blocks_peak=max(
            (used for used, capacity in kv_observed), default=None
        ),
        observed_kv_utilization_peak=max(
            (used / capacity for used, capacity in kv_observed if capacity),
            default=None,
        ),
        observed_scheduled_tokens_peak=max(scheduled_observed, default=None),
        per_request_tok_s=[r["tokens"] / r["latency_s"] for r in results],
        latency_p50_s=quantile([r["latency_s"] for r in results], 0.5),
        latency_p95_s=quantile([r["latency_s"] for r in results], 0.95),
        ttft_p50_s=quantile(
            [r["ttft_s"] for r in results if r["ttft_s"] is not None], 0.5
        ),
        observed_active_peak=max(observed, default=None),
        requests=results,
        peak_vram_mib_sampled=max(vram, default=None),
        vram_sampling_note="nvidia-smi at ~100 ms; may miss shorter allocation peaks",
        queue_wait_s=None,
        queue_wait_note="not exposed per request; TTFT includes admission, prefill and transport",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", choices=["forge", "llama"], required=True)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--out", required=True)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--arrival-gap-ms", type=float, default=0)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--workloads", default="short,medium")
    args = p.parse_args()
    rows = []
    for workload in args.workloads.split(","):
        for concurrency in map(int, args.concurrency.split(",")):
            run(args, concurrency, workload)  # identical full-concurrency warmup
            trials = [run(args, concurrency, workload) for _ in range(args.trials)]
            row = dict(
                concurrency=concurrency,
                workload=workload,
                aggregate_tok_s_median=statistics.median(
                    t["aggregate_tok_s"] for t in trials
                ),
                trials=trials,
            )
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "trials"}), flush=True
            )
            with open(args.out, "w") as f:
                json.dump(
                    dict(
                        runtime=args.runtime,
                        max_new=args.max_new,
                        eos=True,
                        timing="client POST through SSE completion; loopback; load excluded",
                        results=rows,
                    ),
                    f,
                    indent=2,
                )
