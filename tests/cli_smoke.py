"""Opt-in CPU CLI checks with a local pretrained model (Python standard library).

Run after building: python tests/cli_smoke.py --model models/Qwen2.5-0.5B-Instruct
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--cli", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cli = (args.cli or root / "bin" / ("forge-cli.exe" if os.name == "nt" else "forge-cli")).resolve()
    model = args.model.resolve()
    with tempfile.TemporaryDirectory() as cwd:
        def run(flags, text=""):
            return subprocess.run([str(cli), *flags], input=text, cwd=cwd,
                                  capture_output=True, encoding="utf-8", timeout=120)

        for flags in [["-m", "missing"], ["--device", "metal"], ["-c", "0"],
                      ["-n", "-1"], ["--dtype", "bf16"], ["-m"], ["--bogus"]]:
            result = run(flags)
            assert result.returncode != 0 and result.stderr, (flags, result)
        result = run(["-m", str(model), "-c", "256", "-n", "48"],
                     "\n/help\nMy name is Ada.\nWhat is my name?\n/clear\n/exit\n")
        assert result.returncode == 0, result.stderr
        assert result.stdout.count("> ") == 6, result.stdout
        assert "/clear" in result.stdout and "<|im_end|>" not in result.stdout
        result = run(["-m", str(model), "-c", "64", "-n", "2"],
                     " test" * 100 + "\n/clear\nHello!\n/exit\n")
        assert result.returncode == 0 and "Context full:" in result.stderr, result.stderr
        assert result.stdout.replace("> ", "").strip(), result.stdout
        result = run(["-m", str(model), "-c", "256", "-n", "48"],
                     "Repeat exactly: café 日本語 😀\n/exit\n")
        assert result.returncode == 0 and "\ufffd" not in result.stdout, (result.stdout, result.stderr)
    print("CLI flags, repeated turns, blank input, clear, EOS, context recovery and absolute paths: PASS")


if __name__ == "__main__":
    main()
