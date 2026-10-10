"""Real CPU chat HTTP/SSE checks; standard library, optional existing OpenAI SDK."""
import argparse
import concurrent.futures
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = args.model.resolve()
    server = root / "bin" / ("forge-server.exe" if os.name == "nt" else "forge-server")
    runtime_dir = root / "runtime/target/release"
    lib = ctypes.CDLL(str(runtime_dir / ("forge_runtime.dll" if os.name == "nt" else "libforge_runtime.so")))
    lib.forge_runtime_load_ex.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    lib.forge_runtime_load_ex.restype = ctypes.c_void_p
    lib.forge_chat_prepare.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.forge_chat_prepare.restype = ctypes.c_size_t
    lib.forge_runtime_free.argtypes = [ctypes.c_void_p]
    error = ctypes.create_string_buffer(1024)
    rt = lib.forge_runtime_load_ex(str(model).encode(), 256, 0, error, len(error))
    assert rt, error.value

    def prepare(messages):
        data = json.dumps(messages, ensure_ascii=False).encode()
        n = lib.forge_chat_prepare(rt, data, None, 0, error, len(error))
        assert n, error.value
        out = ctypes.create_string_buffer(n)
        assert lib.forge_chat_prepare(rt, data, out, n, error, len(error)) == n
        return json.loads(out.value)

    cases = json.loads((root / "runtime/tests/fixtures/chat_templates.json").read_text(encoding="utf-8"))
    fixture = next(c for c in cases if c["model"] == model.name)
    prepared = prepare(fixture["messages"])
    assert prepared["prompt"] == fixture["formatted"]
    assert prepared["prompt_tokens"] == len(fixture["tokens"])
    conversations = [
        [{"role": "user", "content": "Hello!"}],
        fixture["messages"],
        [{"role": "system", "content": "Reply briefly."}, {"role": "user", "content": "Hello!"}],
        [{"role": "user", "content": "Repeat exactly: café 日本語 😀"}],
        [{"role": "user", "content": ""}],
    ]
    near = [{"role": "user", "content": " test" * 240}]
    while prepare(near)["prompt_tokens"] > 248:
        near[0]["content"] = near[0]["content"][:-5]
    conversations.append(near)
    counts = [prepare(m)["prompt_tokens"] for m in conversations]
    # Count actual sampled IDs independently of the Go scheduler, including EOS.
    class BatchEvent(ctypes.Structure):
        _fields_ = [("request_id", ctypes.c_uint64), ("token_id", ctypes.c_uint32),
                    ("done", ctypes.c_uint8), ("pad", ctypes.c_uint8 * 3),
                    ("piece", ctypes.c_char * 64)]
    lib.forge_batch_admit.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
                                    ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_size_t]
    lib.forge_batch_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.forge_batch_step.restype = ctypes.c_int32
    lib.forge_batch_capacity.argtypes = [ctypes.c_void_p]
    lib.forge_batch_capacity.restype = ctypes.c_size_t
    request_id = ctypes.c_uint64()
    assert lib.forge_batch_admit(rt, prepared["prompt"].encode(), 128, 0,
                                ctypes.byref(request_id), error, len(error)) == 0, error.value
    events = (BatchEvent * lib.forge_batch_capacity(rt))()
    sampled = []
    while True:
        n = lib.forge_batch_step(rt, events, len(events))
        assert n > 0
        sampled.extend(events[i].token_id for i in range(n))
        if any(events[i].done for i in range(n)):
            break
    assert sampled[-1] == prepared["eos_token_id"]
    eos_count = len(sampled)
    lib.forge_runtime_free(rt)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"

    def request(path, body=None, timeout=120):
        data = None if body is None else body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        return urllib.request.urlopen(urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"}), timeout=timeout)

    def complete(body):
        with request("/v1/chat/completions", body) as response:
            assert "Access-Control-Allow-Origin" not in response.headers
            result = json.load(response)
        assert result["object"] == "chat.completion" and result["model"] == api_model
        assert result["id"].startswith("chatcmpl-") and abs(result["created"] - time.time()) < 120
        assert len(result["choices"]) == 1 and result["choices"][0]["message"]["role"] == "assistant"
        text = result["choices"][0]["message"]["content"]
        assert "<|im_" not in text and "�" not in text
        usage = result["usage"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        return result

    def streamed(body, expected):
        with request("/v1/chat/completions", dict(body, stream=True, stream_options={"include_usage": True})) as response:
            assert response.headers["Content-Type"] == "text/event-stream"
            chunks, done = [], False
            for line in response:
                line = line.decode("utf-8").strip()
                if line == "data: [DONE]":
                    done = True
                elif line.startswith("data: "):
                    chunks.append(json.loads(line[6:]))
        assert done and chunks
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert len({c["id"] for c in chunks}) == 1 and len({c["created"] for c in chunks}) == 1
        assert all(c["object"] == "chat.completion.chunk" and c["model"] == api_model for c in chunks)
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        assert text == expected["choices"][0]["message"]["content"], (text, expected)
        assert chunks[-2]["choices"][0]["finish_reason"] == expected["choices"][0]["finish_reason"]
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"] == expected["usage"]

    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    base = {"model": model.name, "messages": conversations[0], "max_tokens": 8, "temperature": 0}
    with tempfile.TemporaryDirectory() as cwd:
        for restart in range(2):
            api_model = model.name if restart == 0 else "smoke-alias"
            base["model"] = api_model
            logpath = Path(cwd) / f"server-{restart}.log"
            with logpath.open("w", encoding="utf-8") as log:
                env = os.environ.copy()
                env["FORGE_RUNTIME_DIR"] = str(runtime_dir)
                proc = subprocess.Popen([str(server), "-model", str(model), "-model-id", api_model, "-max-seq", "256", "-addr", f"127.0.0.1:{port}"], cwd=cwd, env=env, stdout=log, stderr=log, creationflags=flags)
                try:
                    deadline = time.monotonic() + 120
                    while True:
                        assert proc.poll() is None, logpath.read_text(encoding="utf-8")
                        try:
                            with request("/health", timeout=1) as r:
                                assert json.load(r)["status"] == "ok"
                            break
                        except OSError:
                            assert time.monotonic() < deadline, logpath.read_text(encoding="utf-8")
                            time.sleep(0.1)
                    with request("/v1/models") as r:
                        models = json.load(r)
                        assert models["object"] == "list" and [m["id"] for m in models["data"]] == [api_model]
                    expected = complete(base)
                    if restart == 0:
                        for messages, count in zip(conversations, counts):
                            body = dict(base, messages=messages)
                            result = complete(body)
                            assert result["usage"]["prompt_tokens"] == count
                            streamed(body, result)
                        for limit in [1, 2]:
                            result = complete(dict(base, max_tokens=limit))
                            assert result["usage"]["completion_tokens"] == limit and result["choices"][0]["finish_reason"] == "length"
                        eos = complete(dict(base, messages=fixture["messages"], max_tokens=128))
                        assert eos["choices"][0]["finish_reason"] == "stop", eos
                        assert eos["usage"]["completion_tokens"] == eos_count
                        streamed(dict(base, messages=fixture["messages"], max_tokens=128), eos)
                        for patch in [{"model": "wrong"}, {"temperature": 0.7}, {"top_p": 0.5}, {"top_k": 8}, {"n": 2}, {"tools": []}, {"logprobs": True}, {"max_tokens": 0}, {"messages": []}, {"messages": [{"role": "developer", "content": "hi"}]}, {"messages": [{"role": "user", "content": []}]}, {"messages": [{"role": "user", "content": " test" * 300}]}, {"max_tokens": 256}]:
                            for stream in [False, True]:
                                try:
                                    request("/v1/chat/completions", dict(base, **patch, stream=stream)).close()
                                    raise AssertionError(patch)
                                except urllib.error.HTTPError as e:
                                    assert e.code == 400 and "error" in json.load(e)
                        for invalid in [b"{", b"null", b"{} {}", b"\xff"]:
                            try:
                                request("/v1/chat/completions", invalid).close()
                                raise AssertionError(invalid)
                            except urllib.error.HTTPError as e:
                                assert e.code == 400
                        with request("/v1/completions", {"prompt": prepared["prompt"], "max_tokens": 8, "temperature": 0}) as r:
                            legacy = json.load(r)["choices"][0]["text"]
                        assert legacy == complete(dict(base, messages=fixture["messages"]))["choices"][0]["message"]["content"]
                        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                            results = list(pool.map(lambda _: complete(base), range(2)))
                        assert all(r["choices"] == expected["choices"] for r in results)
                        with request("/metrics") as r:
                            before = json.load(r)["requests_cancelled"]
                        with request("/v1/chat/completions", dict(base, stream=True, max_tokens=200, messages=[{"role": "user", "content": "Count from 1 to 1000."}])) as r:
                            r.readline()
                        deadline = time.monotonic() + 30
                        while True:
                            with request("/metrics") as r:
                                metrics = json.load(r)
                            if metrics["requests_cancelled"] > before and metrics["queue_len"] == 0 and not metrics["inference_busy"]:
                                break
                            assert time.monotonic() < deadline, metrics
                            time.sleep(0.1)
                        assert complete(base)["choices"] == expected["choices"]
                        try:
                            import openai
                        except ImportError:
                            print("OpenAI SDK unavailable; standard HTTP checks passed")
                        else:
                            with openai.OpenAI(base_url=url + "/v1", api_key="local-unused") as client:
                                sdk = client.chat.completions.create(**base)
                                assert sdk.choices[0].message.content == expected["choices"][0]["message"]["content"]
                                chunks = list(client.chat.completions.create(**base, stream=True, stream_options={"include_usage": True}))
                                assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == sdk.choices[0].message.content
                                assert chunks[-1].usage.total_tokens == sdk.usage.total_tokens
                            print("OpenAI SDK nonstream/stream: PASS")
                    else:
                        assert expected["choices"] == first["choices"] and expected["usage"] == first["usage"]
                    first = expected
                    active = request("/v1/chat/completions", dict(base, stream=True, max_tokens=200, messages=[{"role":"user","content":"Count from 1 to 1000."}])) if restart else None
                    if active:
                        active.readline()
                    proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
                    if active:
                        with active:
                            failed = active.read().decode("utf-8")
                        assert '"error"' in failed and '[DONE]' not in failed and '"finish_reason":"length"' not in failed and '"finish_reason":"stop"' not in failed, failed
                    assert proc.wait(timeout=40) == 0, logpath.read_text(encoding="utf-8")
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
    print(f"{model.name}: chat HTTP/SSE, templates/counts, empty input, Unicode, context, EOS/length, errors, concurrency, disconnect, restart: PASS")


if __name__ == "__main__":
    main()
