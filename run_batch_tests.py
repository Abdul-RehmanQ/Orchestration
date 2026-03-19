import csv
import subprocess
import sys
import time
from pathlib import Path

TEST_PROMPTS = [
    "make a hello world in python",
    "write a c++ program that adds two numbers",
    "create a function that reverses a string in python",
    "write a c program that prints a multiplication table",
    "make a javascript function that checks if a number is even or odd",

    "build a python script that reads a text file and counts word frequency",
    "write a c++ class for a stack data structure with push pop and peek",
    "create a python function that validates an email address using regex",
    "make a javascript program that sorts an array of objects by age",
    "write a python script that fetches data from a url and prints the response",

    "create a simple login system in python",
    "build a binary search tree in c++",
    "write a rest api client in python that handles authentication",
    "make a task manager program in python with add remove and list features",
    "create a linked list implementation with insert delete and search in c++",

    "write code for sorting",
    "make something that converts celsius to fahrenheit",
    "create an html page with a contact form and css styling",
    "write a typescript interface for a user profile with name age and email",
    "build a concurrent file downloader in go",
]

BACKEND = "qwen"  # or "deepseek"
PYTHON_CMD = sys.executable
TARGET_SCRIPT = "prompt_interface.py"
RESULTS_FILE = "results.txt"
TIMEOUT_SECONDS = 180


def parse_field(line_prefix: str, output_text: str) -> str:
    for line in output_text.splitlines():
        if line.strip().startswith(line_prefix):
            return line.split(":", 1)[1].strip()
    return ""


def run_one(prompt: str) -> dict:
    # Keep each prompt as one CLI argument to preserve punctuation/spacing safely.
    cmd = [PYTHON_CMD, TARGET_SCRIPT, "--backend", BACKEND, prompt]

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            input="\n",
            text=True,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        elapsed = time.perf_counter() - start

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        detected_language = parse_field("Detected language", stdout)
        saved_filename = parse_field("File saved as", stdout)

        status = "ok" if completed.returncode == 0 else "failed"
        error = stderr.strip().replace("\n", " | ")

        return {
            "prompt": prompt,
            "response_time_sec": f"{elapsed:.3f}",
            "file_name": saved_filename,
            "detected_language": detected_language,
            "status": status,
            "error": error,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return {
            "prompt": prompt,
            "response_time_sec": f"{elapsed:.3f}",
            "file_name": "",
            "detected_language": "",
            "status": "timeout",
            "error": f"Timed out after {TIMEOUT_SECONDS}s",
        }


def build_summary(rows: list[dict]) -> dict:
    total = len(rows)
    ok = sum(1 for r in rows if r["status"] == "ok")
    failed = sum(1 for r in rows if r["status"] == "failed")
    timeout = sum(1 for r in rows if r["status"] == "timeout")
    avg_time = (
        sum(float(r["response_time_sec"]) for r in rows) / total if total else 0.0
    )
    return {
        "total": total,
        "ok": ok,
        "failed": failed,
        "timeout": timeout,
        "avg_time_sec": f"{avg_time:.3f}",
    }


def write_results(rows: list[dict], summary: dict) -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt",
                "response_time_sec",
                "file_name",
                "detected_language",
                "status",
                "error",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

        f.write("\n")
        f.write("summary_key\tsummary_value\n")
        f.write(f"total\t{summary['total']}\n")
        f.write(f"ok\t{summary['ok']}\n")
        f.write(f"failed\t{summary['failed']}\n")
        f.write(f"timeout\t{summary['timeout']}\n")
        f.write(f"avg_time_sec\t{summary['avg_time_sec']}\n")


def main() -> None:
    if not Path(TARGET_SCRIPT).exists():
        raise FileNotFoundError(f"{TARGET_SCRIPT} not found in current directory.")

    rows = [run_one(p) for p in TEST_PROMPTS]
    summary = build_summary(rows)
    write_results(rows, summary)

    print(f"Batch complete. Results written to {RESULTS_FILE}")
    print(
        "Summary: "
        f"total={summary['total']}, "
        f"ok={summary['ok']}, "
        f"failed={summary['failed']}, "
        f"timeout={summary['timeout']}, "
        f"avg_time_sec={summary['avg_time_sec']}"
    )


if __name__ == "__main__":
    main()