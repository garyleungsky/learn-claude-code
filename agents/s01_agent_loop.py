#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai",
#   "python-dotenv",
# ]
# ///
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop (OpenRouter version)

The entire secret of an AI coding agent in one pattern:

    while response has tool_calls:
        response = LLM(messages, tools)
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

import json
import os
import subprocess

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL = os.environ.get("MODEL_ID", "openai/gpt-oss-120b:free")

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


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
def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        choice = response.choices[0]
        # Append assistant turn
        messages.append(choice.message.model_dump())
        # If the model didn't call a tool, we're done
        if not choice.message.tool_calls:
            return
        # Execute each tool call, collect results
        for tc in choice.message.tool_calls:
            args = json.loads(tc.function.arguments)
            print(f"\033[33m$ {args['command']}\033[0m")
            output = run_bash(args["command"])
            print(output[:200])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})


if __name__ == "__main__":
    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        last = history[-1]
        if last.get("role") == "assistant" and last.get("content"):
            print(last["content"])
        print()
