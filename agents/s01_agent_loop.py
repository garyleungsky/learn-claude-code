#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "google-genai",
#   "python-dotenv",
# ]
# ///
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop (Gemini version)

The entire secret of an AI coding agent in one pattern:

    while response has function_calls:
        response = LLM(contents, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import subprocess

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(override=True)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = os.environ.get("GEMINI_MODEL_ID", "gemini-2.5-flash")

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = types.Tool(
    function_declarations=[
        {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }
    ]
)

CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM,
    tools=[TOOLS],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(contents: list):
    while True:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=CONFIG,
        )
        # Append model turn
        contents.append(response.candidates[0].content)
        # If the model didn't call a function, we're done
        if not response.function_calls:
            return
        # Execute each function call, collect results
        function_responses = []
        for fc in response.function_calls:
            print(f"\033[33m$ {fc.args['command']}\033[0m")
            output = run_bash(fc.args["command"])
            print(output[:200])
            function_responses.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": output},
                    id=fc.id,
                )
            )
        contents.append(types.Content(role="user", parts=function_responses))


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append(types.Content(role="user", parts=[types.Part(text=query)]))
        agent_loop(history)
        last = history[-1]
        if hasattr(last, "parts"):
            for part in last.parts:
                if hasattr(part, "text") and part.text:
                    print(part.text)
        print()
