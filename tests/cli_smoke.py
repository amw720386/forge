"""Local CPU CLI and HTTP/SSE smoke checks (Python standard library).

Run after building: python tests/cli_smoke.py --model models/Qwen2.5-0.5B-Instruct
"""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import concurrent.futures
import json
import signal
import socket
import time
import urllib.error
import urllib.request
import queue
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--server", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cli = (
        args.cli or root / "bin" / ("forge-cli.exe" if os.name == "nt" else "forge-cli")
    ).resolve()
    server = (
        args.server
        or root / "bin" / ("forge-server.exe" if os.name == "nt" else "forge-server")
    ).resolve()
    model = (root / args.model).resolve()
    for binary in [cli, server]:
        if not binary.is_file():
            raise SystemExit(
                f"Missing {binary}; run scripts/build.ps1 (Windows) or make build (Unix)."
            )
    if not model.exists() or (
        model.is_dir() and not (model / "model.safetensors").is_file()
    ):
        print(f"SKIP model-dependent CLI/HTTP tests: weights absent at {model}")
        return
    cases = json.loads(
        (root / "runtime/tests/fixtures/chat_templates.json").read_text(
            encoding="utf-8"
        )
    )
    family = (
        model.name
        if model.is_dir()
        else (
            "SmolLM2-135M-Instruct"
            if "smollm" in model.name.lower()
            else "Qwen2.5-0.5B-Instruct"
        )
    )
    case = next(c for c in cases if c["model"] == family)
    # Explicit equivalent input: actual qualified template's first user turn.
    formatted = case["formatted"].split("Hi!<|im_end|>", 1)[0]
    flags = ["-m", str(model), "-c", "256", "-n", "8"]
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with tempfile.TemporaryDirectory() as cwd:

        def run(flags, text=""):
            return subprocess.run(
                [str(cli), *flags],
                input=text,
                cwd=cwd,
                capture_output=True,
                encoding="utf-8",
                timeout=120,
            )

        for flags in [
            ["-m", "missing"],
            ["--device", "metal"],
            ["-c", "0"],
            ["-n", "-1"],
            ["--dtype", "bf16"],
            ["-m"],
            ["--bogus"],
        ]:
            result = run(flags)
            assert result.returncode != 0 and result.stderr, (flags, result)
        result = run(
            ["-m", str(model), "-c", "256", "-n", "48"],
            "\n/help\nMy name is Ada.\nWhat is my name?\nWhat name did I give you?\n/clear\n/exit\n",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.count("> ") == 7, result.stdout
        assert "/clear" in result.stdout and "<|im_end|>" not in result.stdout
        result = run(
            ["-m", str(model), "-c", "64", "-n", "2"],
            " test" * 100 + "\n/clear\nHello!\n/exit\n",
        )
        assert result.returncode == 0 and "Context full:" in result.stderr, (
            result.stderr
        )
        assert result.stdout.replace("> ", "").strip(), result.stdout
        result = run(
            ["-m", str(model), "-c", "256", "-n", "48"],
            "Repeat exactly: café 日本語 😀\n/exit\n",
        )
        assert result.returncode == 0 and "\ufffd" not in result.stdout, (
            result.stdout,
            result.stderr,
        )
        flags = ["-m", str(model), "-c", "256", "-n", "8"]
        result = run(flags, "Hello!\n/exit\n")
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("> ") and result.stdout.endswith("\n> "), (
            result.stdout
        )
        expected = result.stdout[2:-3]
        repeat = run(flags, "Hello!\n/clear\nHello!\n/exit\n")
        assert repeat.returncode == 0, repeat.stderr
        assert repeat.stdout == "> " + expected + "\n> > " + expected + "\n> ", (
            repeat.stdout
        )
        # Ctrl+Break on Windows reaches the same interrupt handler as console Ctrl+C.
        proc = subprocess.Popen(
            [str(cli), "-m", str(model), "-c", "256", "-n", "200"],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            creationflags=creationflags,
        )
        try:
            ready = queue.Queue()

            def wait_ready():
                lines = []
                for line in proc.stderr:
                    lines.append(line)
                    if "Ready" in line:
                        ready.put(lines)
                        return
                ready.put(lines)

            reader = threading.Thread(target=wait_ready, daemon=True)
            reader.start()
            lines = ready.get(timeout=120)
            reader.join(timeout=1)
            assert any("Ready" in line for line in lines), lines
            proc.stdin.write("Write every integer from 1 to 1000.\n")
            proc.stdin.flush()
            time.sleep(0.3)
            proc.send_signal(
                signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
            )
            out, err = proc.communicate("/clear\nHello!\n/exit\n", timeout=120)
            assert proc.returncode == 0 and "Generation stopped." in err, (out, err)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        print(
            "CLI: turns, follow-up, clear/reset, help/exit, blank/Unicode/long input, limits, interrupt: PASS"
        )

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        url = f"http://127.0.0.1:{port}"

        def request(body=None, timeout=120):
            data = (
                None
                if body is None
                else (
                    body.encode()
                    if isinstance(body, str)
                    else json.dumps(body).encode()
                )
            )
            req = urllib.request.Request(
                url + ("/health" if data is None else "/v1/completions"),
                data=data,
                headers={"Content-Type": "application/json"},
            )
            return urllib.request.urlopen(req, timeout=timeout)

        body = {"prompt": formatted, "max_tokens": 8, "temperature": 0}
        for restart in range(2):
            with open(
                Path(cwd) / f"server-{restart}.log", "w", encoding="utf-8"
            ) as log:
                env = os.environ.copy()
                env["FORGE_RUNTIME_DIR"] = str(root / "runtime/target/release")
                proc = subprocess.Popen(
                    [
                        str(server),
                        "-model",
                        str(model),
                        "-max-seq",
                        "256",
                        "-addr",
                        f"127.0.0.1:{port}",
                    ],
                    cwd=cwd,
                    env=env,
                    stdout=log,
                    stderr=log,
                    creationflags=creationflags,
                )
                try:
                    deadline = time.monotonic() + 120
                    while True:
                        if proc.poll() is not None:
                            raise AssertionError(
                                (Path(cwd) / f"server-{restart}.log").read_text()
                            )
                        try:
                            with request(timeout=1) as response:
                                assert json.load(response)["status"] == "ok"
                            break
                        except (OSError, urllib.error.URLError):
                            if time.monotonic() > deadline:
                                raise AssertionError("server startup timed out")
                            time.sleep(0.1)

                    def complete():
                        with request(body) as response:
                            return json.load(response)["choices"][0]["text"]

                    assert complete() == expected, (
                        "CLI and HTTP greedy text differ for equivalent formatted input"
                    )
                    if restart == 0:
                        with request(
                            {"prompt": "Hello!", "max_tokens": 1, "temperature": 0}
                        ) as response:
                            assert isinstance(
                                json.load(response)["choices"][0]["text"], str
                            )
                        chunks = []
                        with request(dict(body, stream=True)) as response:
                            assert (
                                response.headers["Content-Type"] == "text/event-stream"
                            )
                            done = False
                            for line in response:
                                line = line.decode("utf-8").strip()
                                if line == "data: [DONE]":
                                    done = True
                                elif line.startswith("data: "):
                                    chunk = json.loads(line[6:])
                                    assert "error" not in chunk, chunk
                                    chunks.append(chunk["choices"][0]["text"])
                            assert done and "".join(chunks) == expected
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=2
                        ) as pool:
                            assert list(pool.map(lambda _: complete(), range(2))) == [
                                expected,
                                expected,
                            ]
                        for invalid in [
                            "{",
                            '{"prompt":"hi"} {}',
                            {"prompt": ""},
                            {"prompt": " "},
                            {"prompt": "hi", "max_tokens": -1},
                            {"prompt": "hi", "temperature": -1},
                            {"prompt": " test" * 300, "max_tokens": 1},
                        ]:
                            try:
                                request(invalid).close()
                                raise AssertionError(
                                    f"accepted invalid input: {invalid}"
                                )
                            except urllib.error.HTTPError as error:
                                assert error.code == 400, (error.code, error.read())
                        with request(
                            {"prompt": " test" * 300, "max_tokens": 1, "stream": True}
                        ) as response:
                            failure = response.read().decode("utf-8")
                            assert (
                                "event: error" in failure
                                and "context limit" in failure
                                and "[DONE]" not in failure
                            ), failure
                        for limit in [1, 2]:
                            with request(dict(body, max_tokens=limit)) as response:
                                assert isinstance(
                                    json.load(response)["choices"][0]["text"], str
                                )
                        # Disconnect after the first streaming event; sibling requests must recover.
                        with request(
                            dict(body, stream=True, max_tokens=200)
                        ) as response:
                            response.readline()
                        try:
                            request(dict(body, max_tokens=200), timeout=0.01).close()
                        except (TimeoutError, socket.timeout, urllib.error.URLError):
                            pass
                        assert complete() == expected
                        deadline = time.monotonic() + 30
                        while True:
                            with urllib.request.urlopen(
                                url + "/metrics", timeout=5
                            ) as response:
                                metrics = json.load(response)
                            if (
                                metrics["requests_cancelled"] > 0
                                and metrics["queue_len"] == 0
                                and not metrics["inference_busy"]
                            ):
                                break
                            assert time.monotonic() < deadline, metrics
                            time.sleep(0.1)
                    active = (
                        request(dict(body, stream=True, max_tokens=200))
                        if restart
                        else None
                    )
                    if active:
                        active.readline()
                    proc.send_signal(
                        signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
                    )
                    if active:
                        with active:
                            active.read()
                    assert proc.wait(timeout=40) == 0
                    assert (
                        "shutting down"
                        in (Path(cwd) / f"server-{restart}.log").read_text()
                    )
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait()
        print(
            "HTTP/SSE: health, greedy parity, streaming, concurrency, cancellation/timeouts, errors, shutdown/restart: PASS"
        )


if __name__ == "__main__":
    main()
