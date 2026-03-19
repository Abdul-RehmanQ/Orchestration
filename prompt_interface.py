import json
import re
import sys
import argparse

import requests
from requests.exceptions import RequestException


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

GENERIC_FILENAMES = {
    "here", "test", "sample", "file", "code", "output", "result", "results",
    "temp", "tmp", "untitled", "main", "app", "script", "program",
    "generated", "generated_logic", "sure", "name",
}

PROMPT_STOPWORDS = {
    "a", "an", "the", "make", "create", "build", "write", "for", "with",
    "and", "or", "to", "of", "in", "on", "based", "page", "file", "code",
    "function", "program",
}

# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

BACKENDS = {
    # Each backend defines endpoint + request protocol used by call_model.
    "deepseek": {
        "url": "http://localhost:11434/api/generate",
        "model": "deepseek-coder:1.3b",
        "type": "ollama",
    },
    "qwen": {
        "url": "http://localhost:8080/v1/chat/completions",
        "model": "qwen",          # model name is ignored by llama-server
        "type": "llamacpp",
    },
}


def call_model(prompt: str, backend: dict) -> str:
    """Send a generation request to the selected backend and return the text response."""
    if backend["type"] == "ollama":
        # Ollama expects a single prompt string at /api/generate.
        payload = {
            "model": backend["model"],
            "prompt": prompt,
            "stream": False,
        }
        # raise_for_status surfaces HTTP failures quickly (bad URL, model missing, etc.).
        response = requests.post(backend["url"], json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")

    elif backend["type"] == "llamacpp":
        # llama-server exposes an OpenAI-compatible /v1/chat/completions endpoint
        payload = {
            "model": backend["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.2,
        }
        response = requests.post(backend["url"], json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    raise ValueError(f"Unknown backend type: {backend['type']}")


def call_classify(code: str, backend: dict) -> dict:
    """Ask the model to classify generated code as structured JSON."""
    prompt = (
        "Classify this code and return ONLY valid JSON with this exact shape:\n"
        "{\"language\":\"...\",\"file_base_name\":\"...\",\"extension\":\"...\"}\n"
        "Rules:\n"
        "- language: programming language name\n"
        "- file_base_name: snake_case, letters/numbers/underscore only, no extension\n"
        "- extension: include leading dot, e.g. .py .cpp .java\n"
        "- Output JSON only (no markdown, no extra text).\n\n"
        f"Code:\n{code}"
    )
    raw = call_model(prompt, backend).strip()

    # Models sometimes wrap JSON in markdown fences; strip those before parsing.
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()
    # If extra prose is returned, salvage the first JSON object via regex.
    if not raw.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            raw = match.group(0)

    return json.loads(raw)


def call_detect_language(code: str, backend: dict) -> str:
    prompt = (
        "Identify the programming language of the following code. "
        "Return only the language name.\n\n"
        f"{code}"
    )
    raw = call_model(prompt, backend)
    normalized = normalize_language_name(raw)
    # If model output is noisy/ambiguous, fall back to lightweight code heuristics.
    if normalized == "txt":
        return infer_language_from_code(code)
    return normalized


def call_detect_filename(code: str, backend: dict) -> str:
    prompt = (
        "Return ONLY a short filename for this code.\n"
        "Rules:\n"
        "- no explanation\n"
        "- no extension\n"
        "- use snake_case\n\n"
        f"{code}"
    )
    raw = call_model(prompt, backend).lower()
    # Keep only a safe snake_case token; default if no valid token is found.
    match = re.search(r"[a-z0-9_]{3,20}", raw)
    return match.group(0) if match else "generated_logic"


# ---------------------------------------------------------------------------
# Filename / language utilities (unchanged from original)
# ---------------------------------------------------------------------------

def sanitize_filename_candidate(value: str) -> str:
    # Normalize user/model filename into portable snake_case without extension.
    candidate = value.strip().lower()
    candidate = re.sub(r"\.[a-z0-9]+$", "", candidate)
    candidate = re.sub(r"[^a-z0-9_]+", "_", candidate)
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    return candidate


def is_low_quality_filename(name: str) -> bool:
    # Reject weak names early to reduce accidental overwrite/confusing outputs.
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
    # Derive a stable filename from meaningful prompt words.
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


def normalize_language_name(raw_language: str) -> str:
    # Clean model text down to a canonical language token.
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
    # Heuristic detector used when model classification is unreliable.
    snippet = code.lower()
    if (
        "#include <iostream>" in snippet or "std::" in snippet
        or "using namespace std" in snippet
        or "cout <<" in snippet or "cin >>" in snippet
    ):
        return "c++"
    if "#include <stdio.h>" in snippet or "printf(" in snippet or "scanf(" in snippet:
        return "c"
    if "def " in snippet and ":" in snippet:
        return "python"
    if "console.log(" in snippet:
        return "javascript"
    if "<html" in snippet or "<!doctype html" in snippet:
        return "html"
    if "{" in snippet and "}" in snippet and ":" in snippet and "<html" not in snippet:
        return "css"
    return "txt"


def validate_classification(
    metadata: dict, code: str, prompt: str, backend: dict
) -> tuple:
    # Start from model output, then aggressively normalize and repair bad values.
    language_raw = str(metadata.get("language", "")).strip()
    file_base_raw = sanitize_filename_candidate(str(metadata.get("file_base_name", "")))
    extension_raw = str(metadata.get("extension", "")).strip().lower()

    normalized_language = normalize_language_name(language_raw)
    if normalized_language == "txt":
        # Secondary language check if first pass is unusable.
        normalized_language = call_detect_language(code, backend)

    inferred_language = infer_language_from_code(code)
    # Prefer deterministic HTML/CSS heuristic over uncertain JS/txt output.
    if inferred_language in {"css", "html"} and normalized_language in {"javascript", "js", "txt"}:
        normalized_language = inferred_language

    expected_ext = LANGUAGE_EXTENSIONS.get(normalized_language, ".txt")

    if is_low_quality_filename(file_base_raw):
        # Try model filename fallback, then prompt-derived fallback if still weak.
        model_fallback_name = sanitize_filename_candidate(call_detect_filename(code, backend))
        if is_low_quality_filename(model_fallback_name):
            file_base = generate_better_filename(prompt, normalized_language)
        else:
            file_base = model_fallback_name
    else:
        file_base = file_base_raw

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
        default="deepseek",
        help="Model backend to use: 'deepseek' (Ollama) or 'qwen' (llama-server on port 8080). Default: deepseek",
    )
    parser.add_argument("prompt", nargs="*", help="Code generation prompt")
    args = parser.parse_args()

    backend = BACKENDS[args.backend]
    print(f"Using backend: {args.backend} ({backend['type']})")

    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        prompt = input("Enter your prompt: ")

    if not prompt.strip():
        print("Prompt cannot be empty.")
        sys.exit(1)

    # Generate code
    try:
        raw_output = call_model(
            f"Write code for: {prompt}. Output code only.",
            backend,
        )
    except RequestException as e:
        # Network/service errors stop execution because output code is unavailable.
        print(f"Request failed: {e}")
        sys.exit(1)

    # Extract fenced code block if present
    code_blocks = re.findall(r"```(?:\w+\n)?(.*?)```", raw_output, re.DOTALL)
    code = code_blocks[0].strip() if code_blocks else raw_output.strip()

    # Classify
    try:
        metadata = call_classify(code, backend)
    except (RequestException, ValueError, json.JSONDecodeError) as e:
        # Fallback path keeps the tool usable when strict JSON classification fails.
        print(f"Classification failed, using fallback: {e}")
        fallback_language = call_detect_language(code, backend)
        metadata = {
            "language": fallback_language,
            "file_base_name": call_detect_filename(code, backend),
            "extension": LANGUAGE_EXTENSIONS.get(fallback_language, ".txt"),
        }

    detected_language, suggested_name, ext = validate_classification(
        metadata, code, prompt, backend
    )

    user_filename = input(f"Enter file name [{suggested_name}]: ").strip()
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

    print(f"Detected language: {detected_language}")
    print(f"File saved as: {filename}")


if __name__ == "__main__":
    main()