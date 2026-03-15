import code
from fileinput import filename
import re
import sys

import requests
from requests.exceptions import RequestException


def normalize_language_name(raw_language: str) -> str:
    """
    Normalize model output into a canonical language label used by extension mapping.
    """
    # Normalize whitespace/case and strip markdown/code-fence wrappers if present.
    text = raw_language.strip().lower()
    text = text.replace("```", " ")
    text = text.replace("`", " ")

    # Remove common prefixes/suffixes such as "language: python" or "python code".
    text = re.sub(r"^(the\s+)?(programming\s+)?language\s*(is|:)\s*", "", text)
    text = re.sub(r"\b(programming\s+language|language|code|snippet|source)\b", "", text)
    text = re.sub(r"[^a-z0-9+#]+", " ", text).strip()

    # Check known aliases in order from more specific to more general.
    alias_map = {
        "c++": "c++",
        "cpp": "cpp",
        "c plus plus": "c++",
        "python": "python",
        "py": "py",
        "javascript": "javascript",
        "js": "js",
        "typescript": "typescript",
        "ts": "ts",
        "golang": "golang",
        "go": "go",
        "rust": "rust",
        "java": "java",
        "php": "php",
        "ruby": "ruby",
        "c": "c",
    }

    if text in alias_map:
        return alias_map[text]

    # If the response is a sentence, use the first matching known token.
    for phrase, canonical in alias_map.items():
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return canonical

    return "txt"


def infer_language_from_code(code: str) -> str:
    """
    Lightweight fallback detection from syntax patterns when model output is noisy.
    """
    snippet = code.lower()

    # C/C++ signatures and directives.
    if "#include" in snippet or "std::" in snippet or "using namespace std" in snippet:
        return "c++"
    if "int main(" in snippet and "printf(" in snippet:
        return "c"

    # Python and common script markers.
    if "def " in snippet and ":\n" in snippet:
        return "python"
    if "import " in snippet and " from " in snippet and "console.log" in snippet:
        return "javascript"

    return "txt"


def detect_language(code: str, url: str) -> str:
    """
    Ask the model to identify the programming language of generated code.
    Returns a lowercase language name (for mapping), or "txt" on failure.
    """
    # Reuse the same model endpoint to classify the language from the code snippet.
    data: dict[str, str | bool] = {
        "model": "deepseek-coder:1.3b",
        "prompt": (
            "Identify the programming language of the following code. "
            "Return only the language name.\n\n"
            f"{code}"
        ),
        "stream": False,
    }

    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        raw_language = result.get("response", "")
        normalized = normalize_language_name(raw_language)

        # If model output cannot be normalized confidently, use code syntax fallback.
        if normalized == "txt":
            return infer_language_from_code(code)

        return normalized
    except Exception:
        # If the request fails, still try code-based inference before defaulting to text.
        inferred = infer_language_from_code(code)
        return inferred if inferred else "txt"

def detect_filename(code: str, url: str) -> str:
    """
    Ask the model to suggest a suitable filename for the generated code.
    Returns filename without extension.
    """
    data: dict[str, str | bool] = {
        "model": "deepseek-coder:1.3b",
        "prompt": (
            "Suggest a short descriptive filename for the following code. "
            "Return only the filename without extension.\n\n"
            f"{code}"
        ),
        "stream": False,
    }

    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        name = result.get("response", "").strip().lower()

        # Remove invalid filename characters
        name = re.sub(r"[^a-z0-9_]", "", name)

        if not name:
            return "generated_logic"

        return name

    except Exception:
        return "generated_logic"

def main() -> None:
    # Accept a prompt from command-line arguments, or fall back to interactive input
    args = sys.argv[1:]
    if args:
        prompt = " ".join(args)
    else:
        prompt = input("Enter your prompt: ")

    # Reject empty or whitespace-only prompts immediately
    if not prompt.strip():
        print("Prompt cannot be empty.")
        sys.exit(1)

    # Ollama local API endpoint
    url = "http://localhost:11434/api/generate"

    # Request payload: instruct the model to return code only
    data: dict[str, str | bool] = {
        "model": "deepseek-coder:1.3b",
        "prompt": f"Write code for: {prompt}. Output code only.",
        "stream": False,
    }

    # Send the request to the Ollama server
    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()  # Raise an error for 4xx/5xx HTTP status codes
        result = response.json()
    except RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(1)
    except ValueError:
        # response.json() raises ValueError when the body is not valid JSON
        print("Server returned a non-JSON response.")
        sys.exit(1)

    # Validate that the expected field is present in the response
    output = result.get("response")
    if output is None:
        print("No 'response' field found in server output.")
        print(result)
        sys.exit(1)

    # Extract the first fenced code block (``` ... ```) if one exists;
    # otherwise use the raw output as the code
    code_blocks = re.findall(r"```(?:\w+\n)?(.*?)```", output, re.DOTALL)
    code = code_blocks[0] if code_blocks else output

    # Use the model itself to detect the language of generated code.
    language = detect_language(code, url)

    # Map detected language names to file extensions.
    # Include common aliases that models often return.
    language_extensions = {
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
    }

    # Default to .txt if the detected language is unknown.
    ext = language_extensions.get(language, ".txt")
# Ask the model to suggest a filename
    suggested_name = detect_filename(code, url)

# Allow user to confirm or modify
    user_filename = input(f"Enter file name [{suggested_name}]: ").strip()

    if not user_filename:
        user_filename = suggested_name

    filename = user_filename + ext

    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"Detected language: {language}")
    print(f"File saved as: {filename}")

if __name__ == "__main__":
    main()