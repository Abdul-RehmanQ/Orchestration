# ---------------------------------------------------------------------------
# prompt_interface.py
#
# AI code generation pipeline with multi-backend support.
# Sends a natural-language prompt to a local LLM, extracts the generated
# code, classifies its language and filename, then saves the file.
#
# Usage:
#   python prompt_interface.py --backend qwen make a python calculator
#   python prompt_interface.py                   (interactive mode)
#
# Backends:
#   qwen     — local llama-server on port 8080 (default, server must be running)
#   deepseek — local Ollama on port 11434
#
# Start llama-server before running this script:
#   C:\llama.cpp\build\bin\llama-server.exe -m <model.gguf> \
#       --n-gpu-layers 999 --ctx-size 2048 --port 8080
# ---------------------------------------------------------------------------

import json
import re
import sys
import argparse
import time

import requests
from requests.exceptions import RequestException


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps canonical language names → file extensions.
# Used to enforce the correct extension after language detection.
LANGUAGE_EXTENSIONS = {
    "python": ".py",
    "py": ".py",
    "c++": ".cpp",
    "cpp": ".cpp",
    "c": ".c",
    "java": ".java",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "go": ".go",
    "golang": ".go",
    "rust": ".rs",
    "php": ".php",
    "ruby": ".rb",
    "html": ".html",
    "css": ".css",
}

# Filenames that are too generic to be useful.
# If the model or fallback logic produces one of these, we replace it.
GENERIC_FILENAMES = {
    "here", "test", "sample", "file", "code", "output", "result", "results",
    "temp", "tmp", "untitled", "main", "app", "script", "program",
    "generated", "generated_logic", "sure", "name",
}

# Words filtered out when generating a filename from the prompt text.
# Keeping only meaningful tokens produces cleaner auto-generated names.
PROMPT_STOPWORDS = {
    "a", "an", "the", "make", "create", "build", "write", "for", "with",
    "and", "or", "to", "of", "in", "on", "based", "page", "file", "code",
    "function", "program",
}

# Maximum characters of generated code forwarded to classification prompts.
# Prevents HTTP 400 errors caused by exceeding the 2048-token context window
# when classifying long outputs (e.g. binary search trees, linked lists).
CODE_SNIPPET_LIMIT = 800

# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

# Each backend entry defines how to reach an inference server.
# "type" controls which request format is used inside call_model().
#   ollama   — POST /api/generate  (Ollama native API)
#   llamacpp — POST /v1/chat/completions  (OpenAI-compatible, used by llama-server)
BACKENDS = {
    "qwen": {
        "url": "http://localhost:8080/v1/chat/completions",
        "model": "qwen",          # model name is ignored by llama-server but required by the schema
        "type": "llamacpp",
    },
    "deepseek": {
        "url": "http://localhost:11434/api/generate",
        "model": "deepseek-coder:1.3b",
        "type": "ollama",
    },
}


# ---------------------------------------------------------------------------
# Utility: code truncation
# ---------------------------------------------------------------------------

def truncate_code(code: str, limit: int = CODE_SNIPPET_LIMIT) -> str:
    """Return the first `limit` characters of code.

    Classification prompts only need enough code to identify the language
    and suggest a filename. Truncating here keeps the combined prompt well
    within the context window without affecting the saved output file.
    """
    return code[:limit]


# ---------------------------------------------------------------------------
# Backend communication
# ---------------------------------------------------------------------------

def call_model(prompt: str, backend: dict) -> str:
    """Send a prompt to the configured backend and return the raw text response.

    Supports two request formats:
    - Ollama native API (/api/generate)
    - OpenAI-compatible API (/v1/chat/completions) used by llama-server and
      cloud providers such as Alibaba Cloud Model Studio.

    Raises requests.exceptions.RequestException on network or HTTP errors.
    """
    if backend["type"] == "ollama":
        payload = {"model": backend["model"], "prompt": prompt, "stream": False}
        response = requests.post(backend["url"], json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")

    elif backend["type"] == "llamacpp":
        # Build headers — add Authorization only when an API key is present
        # (required for cloud backends, not needed for local llama-server).
        headers = {"Content-Type": "application/json"}
        if backend.get("api_key"):
            headers["Authorization"] = f"Bearer {backend['api_key']}"

        payload = {
            "model": backend["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.2,   # low temperature = more deterministic code output
        }
        response = requests.post(
            backend["url"], json=payload, headers=headers, timeout=120
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    raise ValueError(f"Unknown backend type: {backend['type']}")


def call_classify(code: str, backend: dict) -> dict:
    """Ask the model to classify generated code as structured JSON.

    Returns a dict with keys: language, file_base_name, extension.
    Uses a truncated snippet to avoid context window overflow.
    Strips accidental markdown fences the model sometimes adds around JSON.
    """
    snippet = truncate_code(code)
    prompt = (
        "Classify this code and return ONLY valid JSON with this exact shape:\n"
        "{\"language\":\"...\",\"file_base_name\":\"...\",\"extension\":\"...\"}\n"
        "Rules:\n"
        "- language: programming language name\n"
        "- file_base_name: snake_case, letters/numbers/underscore only, no extension\n"
        "- extension: include leading dot, e.g. .py .cpp .java\n"
        "- Output JSON only (no markdown, no extra text).\n\n"
        f"Code:\n{snippet}"
    )
    raw = call_model(prompt, backend).strip()

    # Strip markdown fences if the model wraps its JSON in ```json ... ```
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    # If the model adds preamble text, extract the first JSON object
    if not raw.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            raw = match.group(0)

    return json.loads(raw)


def call_detect_language(code: str, backend: dict) -> str:
    """Fallback: ask the model to identify the programming language by name.

    Called when call_classify() fails or returns an unrecognised language.
    Result is normalised through normalize_language_name(). If normalisation
    still fails, falls back to syntax-based infer_language_from_code().
    """
    snippet = truncate_code(code)
    prompt = (
        "Identify the programming language of the following code. "
        "Return only the language name.\n\n"
        f"{snippet}"
    )
    raw = call_model(prompt, backend)
    normalized = normalize_language_name(raw)
    if normalized == "txt":
        # Model response could not be mapped — use syntax signals instead
        return infer_language_from_code(code)
    return normalized


def call_detect_filename(code: str, backend: dict) -> str:
    """Fallback: ask the model to suggest a short snake_case filename.

    Called when the filename from call_classify() is generic or invalid.
    Returns the first valid token from the model response, or 'generated_logic'
    if no valid token is found.
    """
    snippet = truncate_code(code)
    prompt = (
        "Return ONLY a short filename for this code.\n"
        "Rules:\n"
        "- no explanation\n"
        "- no extension\n"
        "- use snake_case\n\n"
        f"{snippet}"
    )
    raw = call_model(prompt, backend).lower()
    match = re.search(r"[a-z0-9_]{3,20}", raw)
    return match.group(0) if match else "generated_logic"


# ---------------------------------------------------------------------------
# Filename utilities
# ---------------------------------------------------------------------------

def sanitize_filename_candidate(value: str) -> str:
    """Normalise a raw string into a safe snake_case filename base.

    Steps: strip whitespace → lowercase → remove extension → replace
    non-alphanumeric characters with underscores → collapse repeated underscores.
    """
    candidate = value.strip().lower()
    candidate = re.sub(r"\.[a-z0-9]+$", "", candidate)       # strip extension if present
    candidate = re.sub(r"[^a-z0-9_]+", "_", candidate)        # replace invalid chars
    candidate = re.sub(r"_+", "_", candidate).strip("_")      # collapse underscores
    return candidate


def is_low_quality_filename(name: str) -> bool:
    """Return True if the filename is too generic to be meaningful.

    Rejects names that are empty, too short/long, in the GENERIC_FILENAMES
    blacklist, or match a common meaningless pattern like 'main_1'.
    """
    if not re.fullmatch(r"[a-z0-9_]{3,40}", name):
        return True
    if name in GENERIC_FILENAMES:
        return True
    if re.fullmatch(
        r"(file|code|test|sample|output|result|temp|tmp|untitled|main|app|script|program|name)(_[0-9]+)?",
        name,
    ):
        return True
    return False


def generate_better_filename(prompt: str, language: str) -> str:
    """Generate a descriptive filename from the original user prompt.

    Tokenises the prompt, removes stopwords, and joins the first four useful
    tokens with underscores. Falls back to '<language>_generated_file' if no
    useful tokens remain or the result is still low-quality.
    """
    tokens = re.findall(r"[a-z0-9]+", prompt.lower())
    useful_tokens = [t for t in tokens if t not in PROMPT_STOPWORDS and len(t) > 1]
    if useful_tokens:
        base = "_".join(useful_tokens[:4])
    else:
        lang_hint = "cpp" if language in {"c++", "cpp"} else language
        base = f"{lang_hint}_logic"
    base = sanitize_filename_candidate(base)
    if is_low_quality_filename(base):
        lang_hint = "cpp" if language in {"c++", "cpp"} else language
        base = sanitize_filename_candidate(f"{lang_hint}_generated_file")
    return base[:40]


# ---------------------------------------------------------------------------
# Language normalisation
# ---------------------------------------------------------------------------

def normalize_language_name(raw_language: str) -> str:
    """Map a free-text model response to a canonical language label.

    Strips markdown, removes common filler phrases like 'the language is',
    then matches against a known alias map. Returns 'txt' if no match found.

    Examples:
        'The programming language is Python' → 'python'
        'TypeScript'                         → 'typescript'
        'unknown scripting language'         → 'txt'
    """
    text = raw_language.strip().lower()
    text = text.replace("```", " ").replace("`", " ")
    text = re.sub(r"^(the\s+)?(programming\s+)?language\s*(is|:)\s*", "", text)
    text = re.sub(r"\b(programming\s+language|language|code|snippet|source)\b", "", text)
    text = re.sub(r"[^a-z0-9+#]+", " ", text).strip()

    alias_map = {
        "c++": "c++", "cpp": "cpp", "c plus plus": "c++",
        "python": "python", "py": "py",
        "javascript": "javascript", "js": "js",
        "typescript": "typescript", "ts": "ts",
        "golang": "golang", "go": "go",
        "rust": "rust", "java": "java",
        "php": "php", "ruby": "ruby",
        "html": "html", "css": "css", "c": "c",
    }

    if text in alias_map:
        return alias_map[text]
    for phrase, canonical in alias_map.items():
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return canonical
    return "txt"


def infer_language_from_code(code: str) -> str:
    """Detect language from code syntax without using the model (zero network calls).

    Used as a fallback when the model returns an unrecognised language, and as
    a correctness override in validate_classification() for languages that small
    models frequently mislabel (Rust, Go, Java, TypeScript, CSS, HTML).

    Detection order matters: each language is checked before any language whose
    syntax it could be confused with. Key ordering decisions:
    - Rust before TypeScript: both use 'enum', 'struct', type annotations
    - Go before TypeScript: both use braces and short identifiers
    - Java before TypeScript: both use 'interface', 'enum', braces
    - TypeScript before JavaScript: TS is a superset of JS
    - CSS last: brace+colon pattern appears in many languages
    """
    snippet = code.lower()

    # --- Rust ---
    # Distinctive signals: fn, let mut, println!, vec!, pub fn, impl+struct combo
    if (
        "fn main()" in snippet
        or ("fn " in snippet and "->" in snippet)
        or "let mut " in snippet
        or ("struct " in snippet and "impl " in snippet)
        or "use std::" in snippet
        or "println!(" in snippet
        or "vec![" in snippet
        or "pub fn " in snippet
    ):
        return "rust"

    # --- Go ---
    # Distinctive signals: package main, func, fmt., := with func, chan, goroutine
    if (
        "package main" in snippet
        or "func main()" in snippet
        or ("func " in snippet and "fmt." in snippet)
        or (":= " in snippet and "func " in snippet)
        or "go func(" in snippet
        or "chan " in snippet
        or 'import "fmt"' in snippet
    ):
        return "go"

    # --- Java ---
    # Distinctive signals: public class/interface/enum, System.out, import java., @Override
    if (
        "public class " in snippet
        or "public static void main" in snippet
        or "system.out.println" in snippet
        or "import java." in snippet
        or "@override" in snippet
        or "public interface " in snippet
        or "public enum " in snippet
        or "throws " in snippet
    ):
        return "java"

    # --- TypeScript ---
    # Distinctive signals: type annotations (: string, : number), interface, enum,
    # generic <T>, readonly, as-casting. Must come before JavaScript check.
    if (
        ": string" in snippet
        or ": number" in snippet
        or ": boolean" in snippet
        or ": void" in snippet
        or ": any" in snippet
        or ": never" in snippet
        or "interface " in snippet
        or "enum " in snippet
        or "<t>" in snippet
        or "as string" in snippet
        or "as number" in snippet
        or ": string[]" in snippet
        or ": number[]" in snippet
        or "readonly " in snippet
    ):
        return "typescript"

    # --- C++ ---
    # Distinctive signals: iostream include, std:: namespace, cout/cin
    if (
        "#include <iostream>" in snippet or "std::" in snippet
        or "using namespace std" in snippet
        or "cout <<" in snippet or "cin >>" in snippet
    ):
        return "c++"

    # --- C ---
    # Distinctive signals: stdio.h include, printf, scanf
    if "#include <stdio.h>" in snippet or "printf(" in snippet or "scanf(" in snippet:
        return "c"

    # --- Python ---
    # 'def' + ':' is a strong Python signal (functions, methods, classes)
    if "def " in snippet and ":" in snippet:
        return "python"

    # --- JavaScript ---
    # Arrow functions (=>), console.log, const are JS-specific enough at this point
    if "console.log(" in snippet or "=>" in snippet or "const " in snippet:
        return "javascript"

    # --- HTML ---
    if "<html" in snippet or "<!doctype html" in snippet:
        return "html"

    # --- CSS ---
    # Checked last because braces + colons appear in many languages.
    # Extra guards exclude HTML, Python, and JS from triggering this branch.
    if (
        "{" in snippet
        and "}" in snippet
        and ":" in snippet
        and "<html" not in snippet
        and "def " not in snippet
        and "function" not in snippet
    ):
        return "css"

    return "txt"


# ---------------------------------------------------------------------------
# Classification validation
# ---------------------------------------------------------------------------

def validate_classification(metadata: dict, code: str, prompt: str, backend: dict) -> tuple:
    """Validate and correct the LLM classification JSON.

    Pipeline:
    1. Normalise the raw language string from the model response.
    2. If normalisation fails, call the model again for language detection.
    3. Apply syntax-based override for languages small models frequently
       mislabel (Rust, Go, Java, TypeScript, CSS, HTML).
    4. Validate/correct the file extension against LANGUAGE_EXTENSIONS.
    5. Validate the filename; escalate through fallback chain if low-quality.

    Returns (language, file_base_name, extension) as a tuple of strings.
    """
    language_raw = str(metadata.get("language", "")).strip()
    file_base_raw = sanitize_filename_candidate(str(metadata.get("file_base_name", "")))
    extension_raw = str(metadata.get("extension", "")).strip().lower()

    # Step 1+2: normalise language from model output
    normalized_language = normalize_language_name(language_raw)
    if normalized_language == "txt":
        # Model output was unrecognisable — ask again with a direct prompt
        normalized_language = call_detect_language(code, backend)

    # Step 3: syntax-based correctness override.
    # Small models (1-2B parameters) frequently mislabel these six languages.
    # If our syntax rules are confident, override the model's answer.
    inferred_language = infer_language_from_code(code)
    if inferred_language in {"css", "html", "typescript", "rust", "go", "java"} and normalized_language not in {
        "css", "html", "typescript", "ts", "rust", "go", "golang", "java",
    }:
        normalized_language = inferred_language

    # Step 4: derive the correct extension from the validated language
    expected_ext = LANGUAGE_EXTENSIONS.get(normalized_language, ".txt")

    # Step 5: validate filename through three-tier fallback chain
    if is_low_quality_filename(file_base_raw):
        # Tier 1: ask the model for a better name
        model_fallback_name = sanitize_filename_candidate(call_detect_filename(code, backend))
        if is_low_quality_filename(model_fallback_name):
            # Tier 2: generate from prompt tokens (no model call)
            file_base = generate_better_filename(prompt, normalized_language)
        else:
            file_base = model_fallback_name
    else:
        file_base = file_base_raw

    # Enforce the correct extension regardless of what the model returned
    extension = expected_ext if extension_raw != expected_ext else extension_raw
    return normalized_language, file_base, extension


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AI code generator")
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS.keys()),
        default="qwen",
        help="Model backend to use. Default: qwen (local llama-server on port 8080).",
    )
    parser.add_argument("prompt", nargs="*", help="Code generation prompt")
    args = parser.parse_args()

    backend = BACKENDS[args.backend]
    print(f"Using backend: {args.backend} ({backend['type']})")

    # Accept prompt from CLI arguments or fall back to interactive input
    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        prompt = input("Enter your prompt: ")

    if not prompt.strip():
        print("Prompt cannot be empty.")
        sys.exit(1)

    # --- Stage 1: Code generation ---
    start = time.perf_counter()
    try:
        raw_output = call_model(f"Write code for: {prompt}. Output code only.", backend)
    except RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(1)
    elapsed = time.perf_counter() - start

    # Extract the first fenced code block if present; otherwise use the full response
    code_blocks = re.findall(r"```(?:\w+\n)?(.*?)```", raw_output, re.DOTALL)
    code = code_blocks[0].strip() if code_blocks else raw_output.strip()

    # --- Stage 2: Classification ---
    try:
        metadata = call_classify(code, backend)
    except (RequestException, ValueError, json.JSONDecodeError) as e:
        # Classification call failed — build metadata from local fallbacks only
        print(f"Classification failed, using fallback: {e}")
        fallback_language = infer_language_from_code(code)  # syntax-based, no model call
        if fallback_language == "txt":
            try:
                fallback_language = call_detect_language(code, backend)
            except Exception:
                fallback_language = "txt"
        try:
            fallback_name = call_detect_filename(code, backend)
        except Exception:
            fallback_name = generate_better_filename(prompt, fallback_language)
        metadata = {
            "language": fallback_language,
            "file_base_name": fallback_name,
            "extension": LANGUAGE_EXTENSIONS.get(fallback_language, ".txt"),
        }

    # --- Stage 3: Validation ---
    detected_language, suggested_name, ext = validate_classification(
        metadata, code, prompt, backend
    )

    # --- Stage 4: File save ---
    sys.stderr.write(f"Enter file name [{suggested_name}]: ")
    sys.stderr.flush()
    user_filename = sys.stdin.readline().strip()
    if not user_filename:
        user_filename = suggested_name

    user_filename = sanitize_filename_candidate(user_filename)
    if is_low_quality_filename(user_filename):
        better_name = generate_better_filename(prompt, detected_language)
        print(f"Low-quality filename rejected. Using: {better_name}")
        user_filename = better_name

    filename = user_filename + ext
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)

    # Output fields are prefixed so run_batch_tests.py can parse them reliably
    print(f"Detected language: {detected_language}")
    print(f"File saved as: {filename}")
    print(f"Response time: {elapsed:.3f}s")


if __name__ == "__main__":
    main()