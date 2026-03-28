# ---------------------------------------------------------------------------
# run_batch_tests.py
#
# Batch evaluation harness for prompt_interface.py.
#
# What this script measures
# ─────────────────────────
# Previous versions only checked whether the script exited with return code 0
# ("execution success"). That metric is meaningless for research purposes —
# a script that generates Python code but saves it as a .css file still
# returns code 0.
#
# This version adds a real correctness layer:
#   - Every prompt has an EXPECTED language defined up front.
#   - After each run, detected language and file extension are compared to
#     the expected values.
#   - Status is one of: correct | incorrect | system_error | timeout
#   - Summary reports language_accuracy and extension_accuracy as percentages.
#
# Output columns (tab-separated in results.txt):
#   prompt, response_time_sec, file_name, expected_language,
#   detected_language, language_match, extension_match, status, error
# ---------------------------------------------------------------------------

import csv
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Test dataset with expected language labels
# ---------------------------------------------------------------------------
# Each entry is (prompt_string, expected_language).
# expected_language must match a key in LANGUAGE_EXTENSIONS inside
# prompt_interface.py so that extension_match validation works correctly.

TEST_CASES: list[tuple[str, str]] = [
# --- C++ additional ---
    ("write a c++ program that reads a file and prints each line",               "c++"),
    ("implement a stack using arrays in c++",                                    "c++"),
    ("create a c++ template function that returns the maximum of two values",    "c++"),
    ("write a c++ program that sorts a vector of integers using bubble sort",    "c++"),
    ("build a c++ class for a circular queue with enqueue and dequeue",          "c++"),

    # --- Python additional ---
    ("write a python function that flattens a nested list",                      "python"),
    ("create a python class that implements a binary heap",                      "python"),
    ("build a python script that parses command line arguments using argparse",  "python"),
    ("write a python generator that yields fibonacci numbers",                   "python"),
    ("create a python dataclass for a student with name age and grades",         "python"),

    # --- Rust additional ---
    ("write a rust program that reads a csv file line by line",                  "rust"),
    ("create a rust struct for a stack with push pop and peek methods",          "rust"),
    ("implement a rust function that sorts a vector using quicksort",            "rust"),
    ("build a rust program that spawns threads and joins them",                  "rust"),
    ("write a rust enum for a traffic light with red yellow and green states",   "rust"),

    # --- Go additional ---
    ("write a go program that reads environment variables and prints them",      "go"),
    ("create a go interface for an animal with speak and move methods",          "go"),
    ("build a go program that writes json to a file",                            "go"),
    ("implement a go function that reverses a slice of integers",                "go"),
    ("write a go program that reads a text file and counts lines",               "go"),

    # --- Java additional ---
    ("write a java program that sorts an array using bubble sort",               "java"),
    ("create a java class for a binary search tree with insert and search",      "java"),
    ("build a java program that reads user input and checks if it is a palindrome", "java"),
    ("implement a java stack using arraylist",                                   "java"),
    ("write a java program that demonstrates inheritance with animal and dog",   "java"),

    # --- TypeScript additional ---
    ("write a typescript function that debounces another function",              "typescript"),
    ("create a typescript type guard for checking if a value is a string",       "typescript"),
    ("build a typescript class for a linked list with generic type support",     "typescript"),
    ("write a typescript decorator that logs method calls",                      "typescript"),
    ("create a typescript mapped type that makes all properties readonly",       "typescript"),

    # --- JavaScript additional ---
    ("write a javascript function that deep clones an object",                   "javascript"),
    ("create a javascript module that exports utility functions for arrays",     "javascript"),
    ("build a javascript function that throttles another function",              "javascript"),
    ("write a javascript class for a simple pub sub event system",              "javascript"),
    ("implement a javascript function that memoizes another function",           "javascript"),

    # --- CSS additional ---
    ("write css for a sticky navigation bar that stays at the top on scroll",    "css"),
    ("create css styles for a modal dialog with overlay background",             "css"),
    ("build css for a responsive two column layout using grid",                  "css"),
    ("write css for a card component with shadow hover effect",                  "css"),

    # --- HTML additional ---
    ("create an html page with a dropdown menu and submenu items",               "html"),
    ("write an html page that displays a data table with alternating row colors","html"),
    ("build an html registration form with name email password and confirm fields","html"),
    ("create an html page with an image carousel using javascript",              "html"),

    # --- Language ambiguity stress tests ---
    # These prompts do not name the language — forces syntax inference to do all the work
    ("write a function that checks if a number is prime",                        "python"),
    ("implement a binary search algorithm",                                      "python"),
    ("create a class that represents a rectangle with area and perimeter",       "python"),
    ("write a function that converts a string to title case",                    "python"),
    ("build a simple stack data structure",                                      "python"),

    # --- Vague prompts — no expected language, correctness not evaluated ---
    # Expected language set to "" so these rows are excluded from accuracy %
    ("write code",          ""),
    ("make a program",      ""),
    ("do something",        ""),
    ("x",                   ""),
    ("write",               ""),
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND         = "qwen"            # must match a key in BACKENDS inside prompt_interface.py
PYTHON_CMD      = sys.executable    # use the same Python that is running this script
TARGET_SCRIPT   = "prompt_interface.py"
RESULTS_FILE    = "results.txt"
TIMEOUT_SECONDS = 240

# Maps expected language labels to the file extensions we require.
# Must stay in sync with LANGUAGE_EXTENSIONS in prompt_interface.py.
EXPECTED_EXTENSIONS: dict[str, str] = {
    "python":     ".py",
    "c++":        ".cpp",
    "c":          ".c",
    "java":       ".java",
    "javascript": ".js",
    "typescript": ".ts",
    "go":         ".go",
    "rust":       ".rs",
    "html":       ".html",
    "css":        ".css",
}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_field(line_prefix: str, output_text: str) -> str:
    """Extract the value after the first colon on a line that starts with line_prefix.

    prompt_interface.py prints labelled output lines such as:
        Detected language: python
        File saved as: hello_world.py
    This function parses those lines reliably without fragile indexing.
    """
    for line in output_text.splitlines():
        if line.strip().startswith(line_prefix):
            return line.split(":", 1)[1].strip()
    return ""


def get_extension(filename: str) -> str:
    """Return the file extension including the leading dot, e.g. '.py'.

    Returns empty string if no extension is present.
    """
    if "." in filename:
        return "." + filename.rsplit(".", 1)[-1]
    return ""


# ---------------------------------------------------------------------------
# Correctness evaluation
# ---------------------------------------------------------------------------

def evaluate_correctness(
    detected_language: str,
    saved_filename: str,
    expected_language: str,
) -> tuple[bool | None, bool | None]:
    """Compare detected output against expected values.

    Returns (language_match, extension_match) where each value is:
      True  — correct
      False — incorrect
      None  — not evaluated (expected_language is empty, i.e. vague prompt)

    This is the core function that replaces the old binary ok/failed metric.
    """
    if not expected_language:
        # Vague prompts have no defined expected output — skip correctness check
        return None, None

    # Language match: normalise both sides to lowercase for comparison
    lang_match = detected_language.strip().lower() == expected_language.strip().lower()

    # Extension match: derive expected extension from EXPECTED_EXTENSIONS map
    expected_ext = EXPECTED_EXTENSIONS.get(expected_language.lower(), "")
    actual_ext   = get_extension(saved_filename)
    ext_match    = (actual_ext == expected_ext) if expected_ext else None

    return lang_match, ext_match


def assign_status(
    returncode: int,
    language_match: bool | None,
    extension_match: bool | None,
    expected_language: str,
) -> str:
    """Assign a four-value status label to each result row.

    Status values:
      correct      — script ran, language correct, extension correct
      incorrect    — script ran but language or extension was wrong
      system_error — script crashed (non-zero return code)
      timeout      — subprocess timed out
      no_expected  — vague prompt, no expected language defined (not counted in accuracy)
    """
    if returncode != 0:
        return "system_error"

    if not expected_language:
        return "no_expected"   # excluded from accuracy calculation

    if language_match and extension_match:
        return "correct"

    return "incorrect"


# ---------------------------------------------------------------------------
# Single-prompt runner
# ---------------------------------------------------------------------------

def run_one(prompt: str, expected_language: str) -> dict:
    """Run prompt_interface.py for one prompt and return a result dict.

    Passes '\n' as stdin so the filename confirmation prompt auto-accepts
    the suggested name without blocking.
    """
    cmd = [PYTHON_CMD, TARGET_SCRIPT, "--backend", BACKEND, prompt]

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            input="\n",              # auto-confirm the filename prompt
            text=True,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        elapsed = time.perf_counter() - start

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        # Parse labelled output fields from prompt_interface.py stdout
        detected_language = parse_field("Detected language", stdout)
        saved_filename    = parse_field("File saved as",     stdout)

        # Evaluate correctness against expected values
        language_match, extension_match = evaluate_correctness(
            detected_language, saved_filename, expected_language
        )

        status = assign_status(
            completed.returncode, language_match, extension_match, expected_language
        )
        error = stderr.strip().replace("\n", " | ")

        return {
            "prompt":            prompt,
            "response_time_sec": f"{elapsed:.3f}",
            "file_name":         saved_filename,
            "expected_language": expected_language,
            "detected_language": detected_language,
            "language_match":    language_match,    # True / False / None
            "extension_match":   extension_match,   # True / False / None
            "status":            status,
            "error":             error,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return {
            "prompt":            prompt,
            "response_time_sec": f"{elapsed:.3f}",
            "file_name":         "",
            "expected_language": expected_language,
            "detected_language": "",
            "language_match":    None,
            "extension_match":   None,
            "status":            "timeout",
            "error":             f"Timed out after {TIMEOUT_SECONDS}s",
        }


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def build_summary(rows: list[dict]) -> dict:
    """Compute accuracy metrics over all result rows.

    Only rows with a defined expected_language (status != 'no_expected' and
    status != 'timeout' and status != 'system_error') are included in the
    language_accuracy and extension_accuracy calculations.
    """
    total      = len(rows)
    correct    = sum(1 for r in rows if r["status"] == "correct")
    incorrect  = sum(1 for r in rows if r["status"] == "incorrect")
    sys_error  = sum(1 for r in rows if r["status"] == "system_error")
    timed_out  = sum(1 for r in rows if r["status"] == "timeout")
    no_exp     = sum(1 for r in rows if r["status"] == "no_expected")

    # Rows eligible for accuracy calculation (have an expected language and ran)
    evaluable = [r for r in rows if r["expected_language"] and r["status"] != "timeout"]
    n_eval    = len(evaluable)

    lang_correct = sum(1 for r in evaluable if r["language_match"] is True)
    ext_correct  = sum(1 for r in evaluable if r["extension_match"] is True)

    lang_accuracy = (lang_correct / n_eval * 100) if n_eval else 0.0
    ext_accuracy  = (ext_correct  / n_eval * 100) if n_eval else 0.0

    avg_time = (
        sum(float(r["response_time_sec"]) for r in rows) / total if total else 0.0
    )

    return {
        "total":              total,
        "correct":            correct,
        "incorrect":          incorrect,
        "system_error":       sys_error,
        "timeout":            timed_out,
        "no_expected":        no_exp,
        "evaluable":          n_eval,
        "language_accuracy":  f"{lang_accuracy:.1f}%",
        "extension_accuracy": f"{ext_accuracy:.1f}%",
        "avg_time_sec":       f"{avg_time:.3f}",
    }


# ---------------------------------------------------------------------------
# Results file I/O
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "prompt", "response_time_sec", "file_name",
    "expected_language", "detected_language",
    "language_match", "extension_match",
    "status", "error",
]


def init_results_file() -> None:
    """Create the results file and write the header row."""
    with open(RESULTS_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()


def append_result_row(row: dict) -> None:
    """Append one result row to the results file immediately after each run.

    Writing incrementally means partial results are preserved even if the
    batch is interrupted with Ctrl+C.
    """
    with open(RESULTS_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writerow(row)


def append_summary(summary: dict) -> None:
    """Append the summary block at the end of the results file."""
    with open(RESULTS_FILE, "a", encoding="utf-8", newline="") as f:
        f.write("\n")
        f.write("summary_key\tsummary_value\n")
        for key, value in summary.items():
            f.write(f"{key}\t{value}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not Path(TARGET_SCRIPT).exists():
        raise FileNotFoundError(
            f"{TARGET_SCRIPT} not found in the current directory.\n"
            f"Run this script from the same folder as {TARGET_SCRIPT}."
        )

    print(f"Running {len(TEST_CASES)} prompts against backend: {BACKEND}")
    print(f"Target script : {TARGET_SCRIPT}")
    print(f"Results file  : {RESULTS_FILE}")
    print("-" * 70)

    rows: list[dict] = []
    init_results_file()

    try:
        for i, (prompt, expected_lang) in enumerate(TEST_CASES, 1):
            label = f"[{i:02d}/{len(TEST_CASES)}]"
            print(f"{label} {prompt[:55]}")

            result = run_one(prompt, expected_lang)
            rows.append(result)
            append_result_row(result)   # write immediately — safe on Ctrl+C

            # Build a compact one-line status for the console
            lang_ok = result["language_match"]
            ext_ok  = result["extension_match"]
            lang_str = ("✓" if lang_ok else "✗") if lang_ok is not None else "-"
            ext_str  = ("✓" if ext_ok  else "✗") if ext_ok  is not None else "-"

            print(
                f"         → {result['status'].upper():<12} "
                f"lang:{lang_str} ext:{ext_str} | "
                f"{result['detected_language']:<12} | "
                f"{result['file_name']:<30} | "
                f"{result['response_time_sec']}s"
            )

    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial results...")

    finally:
        summary = build_summary(rows)
        append_summary(summary)

        print("-" * 70)
        print(f"Batch complete. Results written to {RESULTS_FILE}")
        print()
        print("=== SUMMARY ===")
        print(f"  Total prompts      : {summary['total']}")
        print(f"  Correct            : {summary['correct']}")
        print(f"  Incorrect          : {summary['incorrect']}")
        print(f"  System errors      : {summary['system_error']}")
        print(f"  Timeouts           : {summary['timeout']}")
        print(f"  No expected (vague): {summary['no_expected']}")
        print(f"  Evaluable prompts  : {summary['evaluable']}")
        print(f"  Language accuracy  : {summary['language_accuracy']}")
        print(f"  Extension accuracy : {summary['extension_accuracy']}")
        print(f"  Avg response time  : {summary['avg_time_sec']}s")


if __name__ == "__main__":
    main()