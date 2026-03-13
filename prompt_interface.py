import re
import sys
import requests
from requests.exceptions import RequestException


def detect_language(code: str, url: str) -> str:
    """
    Ask the model to identify the programming language of the generated code.
    """

    data = {
        "model": "deepseek-coder:1.3b",
        "prompt": f"Identify the programming language of the following code. Return only the language name.\n\n{code}",
        "stream": False,
    }

    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        language = result.get("response", "").strip().lower()
        return language
    except Exception:
        return "txt"


def main() -> None:

    args = sys.argv[1:]

    if args:
        prompt = " ".join(args)
    else:
        prompt = input("Enter your prompt: ")

    if not prompt.strip():
        print("Prompt cannot be empty.")
        sys.exit(1)

    url = "http://localhost:11434/api/generate"

    data = {
        "model": "deepseek-coder:1.3b",
        "prompt": f"Write code for: {prompt}. Output code only.",
        "stream": False,
    }

    try:
        response = requests.post(url, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

    except RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(1)

    except ValueError:
        print("Server returned a non-JSON response.")
        sys.exit(1)

    output = result.get("response")

    if output is None:
        print("No 'response' field found in server output.")
        print(result)
        sys.exit(1)

    code_blocks = re.findall(r"```(?:\w+\n)?(.*?)```", output, re.DOTALL)
    code = code_blocks[0] if code_blocks else output

    # -------- LANGUAGE DETECTION --------

    language = detect_language(code, url)

    language_extensions = {
        "python": ".py",
        "c++": ".cpp",
        "cpp": ".cpp",
        "c": ".c",
        "java": ".java",
        "javascript": ".js",
        "typescript": ".ts",
        "go": ".go",
        "rust": ".rs",
        "php": ".php",
        "ruby": ".rb"
    }

    ext = language_extensions.get(language, ".txt")

    filename = "generated_logic" + ext

    with open(filename, "w") as f:
        f.write(code)

    print(f"Detected language: {language}")
    print(f"Code written to {filename}")


if __name__ == "__main__":
    main()