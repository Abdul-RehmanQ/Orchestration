import csv
import subprocess
import sys
import time
from pathlib import Path

TEST_PROMPTS = [
    # --- Previously failing TypeScript cases (now fixed by infer_language_from_code) ---
    "create a typescript function that fetches and parses json from an api",
    "create a typescript enum for http status codes",
    "write a typescript class for a generic stack data structure",
    "write a typescript interface for a user profile with name age and email",
    "build a typescript utility type that makes all fields of an object optional",

    # --- TypeScript vs JavaScript boundary ---
    "write a javascript arrow function that filters even numbers from an array",
    "create a javascript class with private fields using the hash syntax",
    "write a typescript function that accepts a generic type and returns an array",
    "build a typescript module that exports an interface and a class implementing it",
    "create a javascript async function that retries a failed fetch three times",

    # --- CSS vs JavaScript boundary (previously misclassified) ---
    "write css variables for a dark mode color scheme",
    "create a css flexbox layout for a card grid",
    "write a javascript object that maps color names to hex values",
    "build a css keyframe animation for a bouncing ball",
    "create a javascript function that toggles a css class on a dom element",

    # --- Long C++ (confirmed fixed, keep as regression tests) ---
    "build a binary search tree in c++",
    "create a linked list implementation with insert delete and search in c++",
    "implement a graph with bfs and dfs traversal in c++",
    "write a red black tree implementation in c++",
    "build an lru cache in c++ using a hashmap and doubly linked list",

    # --- Rust (small but distinctive syntax) ---
    "write a rust struct for a point in 2d space with distance method",
    "create a rust function that reads lines from stdin and counts words",
    "build a rust enum for a result type with ok and error variants",
    "write a rust trait for a drawable shape with area and perimeter",
    "implement a rust hashmap that counts character frequency in a string",

    # --- Go concurrency patterns ---
    "write a go program that uses channels to sum numbers concurrently",
    "build a go http middleware that logs request duration",
    "create a go struct with methods for a simple key value store",
    "write a go function that reads a json file into a struct",
    "implement a rate limiter in go using a ticker and channel",

    # --- Java (often confused with C++ by small models) ---
    "write a java generic class for a pair of two values",
    "create a java interface for a shape with area and perimeter methods",
    "build a java program that reads a file line by line and prints each line",
    "write a java enum for days of the week with an is weekend method",
    "implement a java singleton pattern with thread safety",

    # --- Mixed HTML and JS in one file ---
    "create an html page with a button that shows an alert when clicked",
    "build an html page with a javascript countdown timer",
    "write an html form that validates fields with javascript before submit",

    # --- Very short / extreme vague prompts ---
    "code",
    "help",
    "do something",
    "x",
    "write",
]

BACKEND = "qwen"  # or "deepseek"
PYTHON_CMD = sys.executable
TARGET_SCRIPT = "prompt_interface.py"
RESULTS_FILE = "results.txt"
TIMEOUT_SECONDS = 240


def parse_field(line_prefix: str, output_text: str) -> str:
    for line in output_text.splitlines():
        if line.strip().startswith(line_prefix):
            return line.split(":", 1)[1].strip()
    return ""


def run_one(prompt: str) -> dict:
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


def init_results_file() -> None:
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


def append_result_row(row: dict) -> None:
    with open(RESULTS_FILE, "a", encoding="utf-8", newline="") as f:
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
        writer.writerow(row)


def append_summary(summary: dict) -> None:
    with open(RESULTS_FILE, "a", encoding="utf-8", newline="") as f:
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

    print(f"Running {len(TEST_PROMPTS)} prompts against backend: {BACKEND}")
    print(f"Target script: {TARGET_SCRIPT}")
    print("-" * 60)

    rows = []
    init_results_file()

    try:
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"[{i:02d}/{len(TEST_PROMPTS)}] {prompt[:60]}")
            result = run_one(prompt)
            rows.append(result)
            append_result_row(result)
            status_label = result["status"].upper()
            print(f"       -> {status_label} | {result['file_name']} | {result['response_time_sec']}s")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results...")
    finally:
        summary = build_summary(rows)
        append_summary(summary)

        print("-" * 60)
        print(f"Batch complete. Results written to {RESULTS_FILE}")
        print(
            f"Summary: total={summary['total']}, ok={summary['ok']}, "
            f"failed={summary['failed']}, timeout={summary['timeout']}, "
            f"avg_time_sec={summary['avg_time_sec']}"
        )


if __name__ == "__main__":
    main()