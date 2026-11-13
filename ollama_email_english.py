"""
Ollama Email & English Assistant (Part 1 of 3: plain Ollama, no framework)
   Drafts professional emails and corrects English grammar.
   Runs 100% on your computer with a 4-bit quantized model. No external API.

Quick by default. Add --think to see the model's THINKING live (grey);
the RESULT is always printed in one piece at the end.

Run:
    python ollama_email_english.py                          # menu
    python ollama_email_english.py email "ask boss for Friday off"
    python ollama_email_english.py correct "he go to school yesterday"
    python ollama_email_english.py --think correct "..."     # show thinking (slower)
    python ollama_email_english.py -m llama3.2:3b ...        # pick a model
"""

import os
import subprocess
import sys

# If started with another Python, re-run with the .venv next to this file
# (or one folder up), so plain "python this_file.py" always works.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BIN = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
for _VENV in (os.path.join(_HERE, ".venv"), os.path.join(os.path.dirname(_HERE), ".venv")):
    _VENV_PY = os.path.join(_VENV, *_BIN)
    if __name__ == "__main__" and os.path.exists(_VENV_PY):
        if os.path.normcase(sys.prefix) != os.path.normcase(_VENV):
            sys.exit(subprocess.call([_VENV_PY, *sys.argv]))
        break

import argparse
import time

import ollama

# All 4-bit quantized (Q4_K_M). The first one you have installed is used.
MODELS = ["qwen3.5:4b", "llama3.2:3b", "qwen2.5:3b", "qwen3.5:latest"]

os.system("")  # enable colours in the Windows console
GREY, BOLD, RESET = "\033[90m", "\033[1m", "\033[0m"

EMAIL_PROMPT = (
    "You write short, polite, professional emails. Output only a 'Subject:' line, "
    "then the email with greeting and sign-off. Keep the body to 2-4 short sentences. "
    "Use ONLY the facts the user gives. Do not add promises, reasons or details "
    "(no lines like 'I will finish my work first' or 'I will be available by email'). "
    "Use [placeholders] for unknown dates or names."
)
CORRECT_PROMPT = (
    "You are a careful English proofreader. Fix grammar, spelling, tense and word choice. "
    "Keep the meaning and tone. Reply in exactly this format:\n"
    "Corrected: <the full corrected text>\n"
    "Changes:\n"
    "- <old> -> <new> (short reason)\n"
    "List only words you actually changed.\n\n"
    "Example for 'She goed to the market.':\n"
    "Corrected: She went to the market.\n"
    "Changes:\n"
    "- goed -> went (irregular past tense)\n\n"
    "Example for 'Me and him was late.':\n"
    "Corrected: He and I were late.\n"
    "Changes:\n"
    "- Me and him -> He and I (subject pronouns)\n"
    "- was -> were (plural subject)"
)


def pick_model() -> str:
    installed = [m.model for m in ollama.list().models]
    return next((m for m in MODELS if m in installed), MODELS[0])


def can_think(model: str) -> bool:
    return "thinking" in (ollama.show(model).capabilities or [])


def ask(model: str, think: bool, system: str, text: str) -> str:
    """Show the thinking live, then print the whole answer at once."""
    start = time.time()
    stream = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
        think=think,
        options={"temperature": 0.3,
                 "num_ctx": 4096 if think else 2048,          # smaller = faster
                 "num_predict": 2500 if think else 500},
        stream=True,
        keep_alive="30m",                             # stay in RAM = quick next answer
    )

    answer = ""
    thinking = False
    for chunk in stream:
        thought = chunk.message.thinking or ""
        if thought:                                   # thinking -> print now
            if not thinking:
                print(f"{GREY}----- thinking -----")
                thinking = True
            print(thought, end="", flush=True)
        answer += chunk.message.content or ""         # answer -> keep for later
    if thinking:
        print(f"\n----- thinking done -----{RESET}")

    print(f"\n{BOLD}===== RESULT ====={RESET}\n{answer.strip()}")
    print(f"{GREY}({time.time() - start:.0f}s){RESET}")
    return answer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", nargs="?", choices=["email", "correct"])
    p.add_argument("text", nargs="?")
    p.add_argument("-m", "--model", help="Ollama model name")
    p.add_argument("--think", action="store_true", help="show the model's thinking (slower)")
    args = p.parse_args()

    model = args.model or pick_model()
    think = can_think(model) and args.think
    print(f"[plain ollama | model: {model} | thinking: {'on' if think else 'off'}]")
    ollama.generate(model=model, prompt="", keep_alive="30m")  # load into RAM now
    prompts = {"email": EMAIL_PROMPT, "correct": CORRECT_PROMPT}

    if args.task and args.text:
        ask(model, think, prompts[args.task], args.text)
        return

    while True:
        choice = input("\n1) Draft email   2) Correct English   q) Quit\n> ").strip().lower()
        if choice == "q":
            break
        if choice == "1":
            text = input("What should the email say?\n> ")
            ask(model, think, EMAIL_PROMPT, text)
        elif choice == "2":
            text = input("Text to correct:\n> ")
            ask(model, think, CORRECT_PROMPT, text)
        else:
            print("Please type 1, 2 or q.")


if __name__ == "__main__":
    main()
