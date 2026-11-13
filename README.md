# Part 1: Local AI Writing Assistant with Plain Ollama

> **Series:** **Part 1: Ollama** · [Part 2: LangChain](https://github.com/rizwannazir333/langchain-email-english) · [Part 3: LangGraph](https://github.com/rizwannazir333/langgraph-email-english)

Build an **email writer** and an **English grammar checker** that run **100% on your own computer**.
No API key, no internet, no cost.

This is **Part 1** of a 3-part series. Here we use **plain Ollama** with no framework, so you can see
exactly how a program talks to an AI model. In Parts 2 and 3 we build the *same app* with LangChain
and LangGraph to see what those tools add.

**What you will learn:**
- How to run an AI model locally with Ollama
- How to send **system** and **user** messages to a model
- How to **stream** an answer and show the model's **thinking**
- How **few-shot examples** in a prompt improve small models

---

## 1. What you need

- A computer with **8 GB RAM or more** (16 GB recommended). **No graphics card needed.**
- **Python 3.10 or newer** ([python.org](https://www.python.org/downloads/))
- **Ollama**, a free program that runs AI models on your computer ([ollama.com/download](https://ollama.com/download))
- About **4 GB of free disk space** for the model

---

## 2. Setup (10 minutes)

**Step 1: download the AI model.** After installing Ollama, open a terminal:

```bash
ollama pull qwen3.5:4b
ollama list          # check it is installed
```

**Step 2: get the code.**

```bash
git clone https://github.com/rizwannazir333/ollama-email-english.git
cd ollama-email-english
```

**Step 3: create a virtual environment and install the library.**

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Mac / Linux:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**Step 4: run it.**

```bash
python 1_ollama_email_english.py
```

```
[plain ollama | model: qwen3.5:4b | thinking: off]

1) Draft email   2) Correct English   q) Quit
>
```

Type `2`, press Enter, then type `she dont like coffee`. After about 20 seconds:

```
===== RESULT =====
Corrected: She doesn't like coffee.
Changes:
- dont -> doesn't (third person singular verb)
- she -> She (sentence case capitalization)
```

> **Good to know:** the script finds the `.venv` folder by itself, so plain `python 1_ollama_email_english.py`
> works even if you forget to activate the environment.

---

## 3. Key ideas

| Word | Simple meaning |
|---|---|
| **LLM** | *Large Language Model*. A program that reads text and writes text. |
| **Ollama** | A free app that downloads LLMs and runs them **locally** (on your computer). |
| **Quantized model** | A model whose numbers are stored with fewer bits (here 4-bit, `Q4_K_M`). It becomes **~4× smaller and faster**, with only a small loss in quality. That is why a 4-billion-parameter model fits in 3.3 GB. |
| **Prompt** | The instructions we give the model. |
| **System message** | The "job description" for the model, e.g. *"You are a careful English proofreader."* |
| **User message** | What the user typed, e.g. *"she dont like coffee"*. |
| **Thinking** | Some models (like Qwen 3.5) can "think out loud" before answering. More accurate, but **much slower** on a CPU. |
| **Token** | A small piece of a word. Models read and write tokens. Fewer tokens means a faster answer. |

---

## 4. The lesson: how the code works

Open [`1_ollama_email_english.py`](1_ollama_email_english.py). It is only about 140 lines.

### Step A: two prompts, one for each job

```python
EMAIL_PROMPT   = "You write short, polite, professional emails. ..."
CORRECT_PROMPT = "You are a careful English proofreader. ..."
```

The **same code** does both jobs; only the **system prompt** changes. This is the most important idea
in prompt engineering: *the prompt is the program*.

### Step B: few-shot examples

The correction prompt includes worked examples:

```
Example for 'She goed to the market.':
Corrected: She went to the market.
Changes:
- goed -> went (irregular past tense)
```

Small models copy examples very well. Showing the model **what a good answer looks like** is called
**few-shot prompting**, and it often works better than long instructions.

### Step C: send messages to the model

```python
stream = ollama.chat(
    model=model,
    messages=[{"role": "system", "content": system},   # the job description
              {"role": "user", "content": text}],      # what the user typed
    think=think,                                         # thinking on or off
    options={"temperature": 0.3, ...},                   # low = more focused answers
    stream=True,                                         # get the answer piece by piece
    keep_alive="30m",                                    # keep the model in memory
)
```

### Step D: show thinking live, print the answer at once

```python
for chunk in stream:
    thought = chunk.message.thinking or ""
    if thought:
        print(thought, end="", flush=True)        # thinking -> print now (grey)
    answer += chunk.message.content or ""         # answer   -> collect for later
...
print("===== RESULT =====")
print(answer)                                     # the full answer, in one piece
```

Try it with thinking on: `python 1_ollama_email_english.py --think correct "me and him is late"`.
You will see the model reason step by step (slowly!) before the result.

### What plain Ollama *can't* do easily

Everything here is manual. There is **no memory** (it forgets the last email), **no tools** (it can't
look up today's date), and the answer is **plain text** that is hard to use in code.
👉 That is exactly what **[Part 2: LangChain](https://github.com/rizwannazir333/langchain-email-english)** adds.

---

## 5. Command options

```bash
python 1_ollama_email_english.py                                   # menu
python 1_ollama_email_english.py email "ask my manager for Friday off"
python 1_ollama_email_english.py correct "he go to school yesterday"
python 1_ollama_email_english.py --think correct "..."             # show thinking (slower)
python 1_ollama_email_english.py -m llama3.2:3b correct "..."      # use another model
```

| Option | Meaning |
|---|---|
| `--think` | Show the model's thinking live (more accurate, **much slower**) |
| `-m MODEL` | Use another Ollama model, e.g. `-m llama3.2:3b` |

If you don't choose a model, the script uses the **first installed one** from this list:
`qwen3.5:4b` → `llama3.2:3b` → `qwen2.5:3b` → `qwen3.5:latest`.

---

## 6. Making it fast on a normal laptop

Tested on a laptop with **no graphics card** (Intel i7, 16 GB RAM): **about 15–30 seconds per answer**.

| Trick | Why it helps |
|---|---|
| **Quantized 4B model** | Small enough for a CPU, still good at grammar |
| **Thinking off by default** | Thinking can take *minutes* on a CPU |
| **Small context** (`num_ctx=2048`) | Less memory to process means faster answers |
| **Short answers** (`num_predict`) | The model stops sooner |
| **`keep_alive="30m"`** + loading at start-up | The model stays in memory, so no waiting to load |

> **Honest note:** small models make mistakes. They sometimes add a sentence you didn't ask for
> (*"I will ensure my tasks are completed"*). **Always read an AI email before you send it.**

---

## 7. Exercises

1. Add a third menu option: **"Summarise text"** with its own system prompt.
2. Add a new example to `CORRECT_PROMPT`, e.g. *"I has two cat."* → *"I have two cats."*
3. Make every email end with *"Kind regards"*.
4. Change `temperature` from `0.3` to `1.0`. Run the same email 3 times. What changes?
5. Compare `qwen3.5:4b` and `llama3.2:3b` on 5 sentences. Which is faster? Which is more accurate?

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `model "qwen3.5:4b" not found` | Run `ollama pull qwen3.5:4b` |
| `Connection refused` / cannot connect | Start the Ollama app (or run `ollama serve`) |
| `ModuleNotFoundError: No module named 'ollama'` | Do [Setup step 3](#2-setup-10-minutes) |
| The menu seems to do nothing | Type the **number** first (`1` or `2`), press Enter, *then* your text |
| Very slow (minutes) | Close other programs, don't use `--think`, check `ollama ps` |
| Strange codes like `[90m` | Use Windows Terminal or the VS Code terminal (they support colours) |

---

## Files

```
.
├── 1_ollama_email_english.py   # the app
├── requirements.txt            # Python library (tested version)
├── .gitignore                  # keeps .venv and cache files out of Git
└── README.md                   # this lesson
```

**Next:** [Part 2: the same app with LangChain](https://github.com/rizwannazir333/langchain-email-english)
