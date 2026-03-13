import re
import sys

import requests
from requests.exceptions import RequestException


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

    # Write the generated code to a file for later execution
    with open("generated_logic.py", "w") as f:
        f.write(code)

    print("Code written to generated_logic.py")


if __name__ == "__main__":
    main()