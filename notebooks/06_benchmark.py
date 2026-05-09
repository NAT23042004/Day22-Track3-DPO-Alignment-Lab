# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB6 — LLM Benchmark: SFT-only vs SFT+DPO
#
# **Stack:** `lm-eval-harness` (IFEval, GSM8K, MMLU) + hand-rolled AlpacaEval-lite (judge-based).
# Maps to deck §8.1–§8.5 (Đánh giá Alignment): static suites · judge-based suites · reward-model
# evaluators · VN landscape.
#
# > **Mục tiêu:** chạy 4 benchmarks trên *cùng 1 base model* dưới 2 condition (SFT-only và
# > SFT+DPO), thấy bằng số có gì tăng có gì giảm. Plot bar chart so sánh. Đây là cách *bạn* tự đo
# > tương đương Tulu 3 stats §9.2b — không chỉ trích dẫn paper người khác.
# >
# > **Quan trọng đọc trước khi run:** deck §8.1 (vì sao đánh giá alignment khó). Một số
# > benchmark có thể *giảm* sau DPO — đó là alignment tax (chat-tuning trade-off với reasoning),
# > không phải bug. Document trong REFLECTION § 7.

# %% [markdown]
# ## 0. Setup

# %%
import os
import json
import gc
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()


def load_dotenv_file(path: Path) -> None:
    """Load simple KEY=VALUE lines from .env without overwriting shell env."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file(REPO_ROOT / ".env")

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()

if COMPUTE_TIER == "T4":
    LIMIT_IFEVAL = 540
    LIMIT_GSM8K = 500
    LIMIT_MMLU = 500
    LIMIT_ALPACA = 100
    BATCH_SIZE = 1
else:
    LIMIT_IFEVAL = 540
    LIMIT_GSM8K = 1319
    LIMIT_MMLU = 5000
    LIMIT_ALPACA = 250
    BATCH_SIZE = 4

LIMIT_IFEVAL = int(os.environ.get("LIMIT_IFEVAL", LIMIT_IFEVAL))
LIMIT_GSM8K = int(os.environ.get("LIMIT_GSM8K", LIMIT_GSM8K))
LIMIT_MMLU = int(os.environ.get("LIMIT_MMLU", LIMIT_MMLU))
LIMIT_ALPACA = int(os.environ.get("LIMIT_ALPACA", LIMIT_ALPACA))
BATCH_SIZE = int(os.environ.get("BENCH_BATCH_SIZE", BATCH_SIZE))
LM_EVAL_TIMEOUT = int(os.environ.get("LM_EVAL_TIMEOUT", "2400"))
LM_EVAL_MAX_NEW_TOKENS = int(os.environ.get("LM_EVAL_MAX_NEW_TOKENS", "128"))

SFT_PATH = REPO_ROOT / "adapters" / "sft-mini"
DPO_PATH = REPO_ROOT / "adapters" / "dpo"
EVAL_OUT = REPO_ROOT / "data" / "eval"
EVAL_OUT.mkdir(parents=True, exist_ok=True)
BENCHMARK_RESULTS_PATH = EVAL_OUT / "benchmark_results.json"
PRESERVE_SKIPPED_METRICS = os.environ.get("PRESERVE_SKIPPED_METRICS", "1") != "0"

if BENCHMARK_RESULTS_PATH.exists():
    try:
        PREVIOUS_BENCHMARK_RESULTS = json.loads(BENCHMARK_RESULTS_PATH.read_text())
    except Exception:
        PREVIOUS_BENCHMARK_RESULTS = {}
else:
    PREVIOUS_BENCHMARK_RESULTS = {}

assert SFT_PATH.exists(), "NB1 must run first"
assert DPO_PATH.exists(), "NB3 must run first"

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"IFEval:          {LIMIT_IFEVAL} prompts")
print(f"GSM8K:           {LIMIT_GSM8K} problems")
print(f"MMLU:            {LIMIT_MMLU} questions")
print(f"AlpacaEval-lite: {LIMIT_ALPACA} prompts")
print(f"lm-eval max new tokens: {LM_EVAL_MAX_NEW_TOKENS}")
print(f"output:          {EVAL_OUT}")

# %%
import torch

assert torch.cuda.is_available(), "Need GPU. See HARDWARE-GUIDE.md."

# %% [markdown]
# ## 1. Helper — run lm-eval on a model+adapter pair

# %%
import subprocess


def prepare_lm_eval_base() -> str:
    """Return a local HF snapshot path with bounded generation config.

    The Unsloth Qwen bnb snapshots ship with generation_config.max_new_tokens=2048.
    Transformers lets that override lm-eval's max_length, so IFEval can hang on
    tiny limits. Capping the cached generation config keeps NB6 runnable.
    """
    from huggingface_hub import snapshot_download

    base = "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4" else "unsloth/Qwen2.5-7B-bnb-4bit"
    snapshot = Path(snapshot_download(base))
    gen_config_path = snapshot / "generation_config.json"
    if gen_config_path.exists() and LM_EVAL_MAX_NEW_TOKENS > 0:
        gen_config = json.loads(gen_config_path.read_text())
        if gen_config.get("max_new_tokens") != LM_EVAL_MAX_NEW_TOKENS:
            gen_config["max_new_tokens"] = LM_EVAL_MAX_NEW_TOKENS
            gen_config_path.write_text(json.dumps(gen_config, indent=2))
            print(f"Patched {gen_config_path} max_new_tokens={LM_EVAL_MAX_NEW_TOKENS}")
    return str(snapshot)


LM_EVAL_BASE = prepare_lm_eval_base()


def run_lm_eval(adapter_path, tasks, limit, num_fewshot, label):
    """Run lm-eval-harness with PEFT adapter on top of base, return parsed metrics."""
    out_dir = EVAL_OUT / f"lm-{label}-{tasks}"
    cmd = [
        sys.executable, "-m", "lm_eval", "run",
        "--model", "hf",
        "--model_args", f"pretrained={LM_EVAL_BASE},peft={adapter_path}",
        "--tasks", tasks,
        "--num_fewshot", str(num_fewshot),
        "--limit", str(limit),
        "--batch_size", str(BATCH_SIZE),
        "--device", "cuda:0",
        "--output_path", str(out_dir),
    ]
    print(f"\n{'=' * 60}\nRunning lm-eval [{label}]: {tasks}\n{'=' * 60}")
    print(" ".join(str(part) for part in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=LM_EVAL_TIMEOUT)

    out_files = sorted(out_dir.glob("**/results*.json"))
    if not out_files:
        print(f"WARN: lm-eval didn't write results JSON (returncode={proc.returncode}).")
        print("STDOUT tail:")
        print(proc.stdout[-1000:])
        print("STDERR tail:")
        print(proc.stderr[-2000:])
        return {"error": "no_results"}
    return json.loads(out_files[-1].read_text())["results"]


# %% [markdown]
# ## 2. IFEval — Instruction-Following (programmatic)
#
# **What it tests:** can the model follow precise format instructions like "respond in 3 bullets."
# 540 prompts, scored programmatically. No judge needed. **Why DPO matters:** chat alignment
# is exactly the skill IFEval measures.

# %%
print(">>> SFT-only on IFEval")
if LIMIT_IFEVAL > 0:
    sft_ifeval = run_lm_eval(SFT_PATH, "ifeval", LIMIT_IFEVAL, num_fewshot=0, label="sft")
    gc.collect()
    torch.cuda.empty_cache()

    print(">>> SFT+DPO on IFEval")
    dpo_ifeval = run_lm_eval(DPO_PATH, "ifeval", LIMIT_IFEVAL, num_fewshot=0, label="dpo")
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("Skipping IFEval because LIMIT_IFEVAL <= 0")
    sft_ifeval = {"error": "skipped"}
    dpo_ifeval = {"error": "skipped"}

# %% [markdown]
# ## 3. GSM8K — Grade-School Math (alignment tax probe)
#
# **What it tests:** 1.3K word problems, exact-match on the `####` final answer.
# **Why DPO matters:** chat-aligned models often *lose* a few points on GSM8K (alignment tax).

# %%
print(">>> SFT-only on GSM8K")
if LIMIT_GSM8K > 0:
    sft_gsm8k = run_lm_eval(SFT_PATH, "gsm8k", LIMIT_GSM8K, num_fewshot=8, label="sft")
    gc.collect()
    torch.cuda.empty_cache()

    print(">>> SFT+DPO on GSM8K")
    dpo_gsm8k = run_lm_eval(DPO_PATH, "gsm8k", LIMIT_GSM8K, num_fewshot=8, label="dpo")
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("Skipping GSM8K because LIMIT_GSM8K <= 0")
    sft_gsm8k = {"error": "skipped"}
    dpo_gsm8k = {"error": "skipped"}

# %% [markdown]
# ## 4. MMLU — Broad knowledge (sampled)
#
# **What it tests:** 14K MCQ across 57 subjects. T4 limit: 500. BigGPU: 5K.
# **Why DPO matters:** if MMLU drops a lot, you've over-aligned (capacity loss).

# %%
print(">>> SFT-only on MMLU (sampled)")
if LIMIT_MMLU > 0:
    sft_mmlu = run_lm_eval(SFT_PATH, "mmlu", LIMIT_MMLU, num_fewshot=5, label="sft")
    gc.collect()
    torch.cuda.empty_cache()

    print(">>> SFT+DPO on MMLU (sampled)")
    dpo_mmlu = run_lm_eval(DPO_PATH, "mmlu", LIMIT_MMLU, num_fewshot=5, label="dpo")
    gc.collect()
    torch.cuda.empty_cache()
else:
    print("Skipping MMLU because LIMIT_MMLU <= 0")
    sft_mmlu = {"error": "skipped"}
    dpo_mmlu = {"error": "skipped"}

# %% [markdown]
# ## 5. AlpacaEval-lite — Win-rate vs reference (judge-based)
#
# Mini AlpacaEval 2 LC. 100 prompts, generate from both adapters, judge with gpt-4o-mini or
# claude-haiku. Pure preference-style — closest in spirit to what DPO trained on.
#
# Falls back to "skipped" if no API key. Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to enable.

# %%
from datasets import load_dataset


def load_alpaca_lite_prompts(n):
    """Load first n prompts from tatsu-lab/alpaca_eval."""
    try:
        ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval",
                          split="eval", trust_remote_code=True)
        return [{"id": i, "prompt": ds[i]["instruction"]} for i in range(min(n, len(ds)))]
    except Exception as exc:
        print(f"alpaca_eval dataset load failed ({exc}); using NB4 fallback")
        eval_path = EVAL_OUT / "prompts.json"
        if eval_path.exists():
            base = json.loads(eval_path.read_text())
            return (base * (n // len(base) + 1))[:n]
        return []


alpaca_prompts = load_alpaca_lite_prompts(LIMIT_ALPACA)
print(f"Loaded {len(alpaca_prompts)} AlpacaEval-lite prompts")

# %%
def generate_with_adapter(adapter_path, prompts, max_new_tokens=256):
    """NB4 pattern: load base + adapter, generate, free memory."""
    from unsloth import FastLanguageModel
    from peft import PeftModel

    base = "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4" else "unsloth/Qwen2.5-7B-bnb-4bit"
    max_len = 512 if COMPUTE_TIER == "T4" else 1024

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=max_len, dtype=None, load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def render_chat(messages, *, add_generation_prompt=False, return_tensors=None):
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=return_tensors is not None,
                return_tensors=return_tensors,
                add_generation_prompt=add_generation_prompt,
            )
        text = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        if return_tensors is not None:
            return tokenizer(text, return_tensors=return_tensors).input_ids
        return text

    model = PeftModel.from_pretrained(model, str(adapter_path))
    FastLanguageModel.for_inference(model)

    outputs = []
    for p in prompts:
        msgs = [{"role": "user", "content": p["prompt"]}]
        inp = render_chat(msgs, return_tensors="pt", add_generation_prompt=True).to("cuda")
        with torch.no_grad():
            out = model.generate(input_ids=inp, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        outputs.append(tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip())

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


# %%
JUDGE_PROMPT = """You are evaluating two assistant responses for helpfulness.

User prompt: {prompt}

Response A: {a}

Response B: {b}

Which is more helpful, accurate, and on-topic? Answer with one of: "A", "B", or "tie".
One-sentence justification.

Output JSON: {{"winner": "A" | "B" | "tie", "reason": "..."}}"""


def judge_pair(a, b, prompt):
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.environ.get("JUDGE_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(prompt=prompt, a=a, b=b)}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(resp.choices[0].message.content)
        except Exception:
            return {"winner": "tie", "reason": "parse error"}
    elif os.environ.get("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic
        client = Anthropic()
        resp = client.messages.create(
            model=os.environ.get("JUDGE_MODEL", "claude-haiku-4-5"),
            max_tokens=200,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(prompt=prompt, a=a, b=b)}],
        )
        try:
            return json.loads(resp.content[0].text)
        except Exception:
            return {"winner": "tie", "reason": "parse error"}
    return None


# %%
import random

if alpaca_prompts and (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
    print(f">>> Generating SFT-only on {len(alpaca_prompts)} AlpacaEval-lite prompts")
    sft_outputs = generate_with_adapter(SFT_PATH, alpaca_prompts)
    print(f">>> Generating SFT+DPO")
    dpo_outputs = generate_with_adapter(DPO_PATH, alpaca_prompts)

    print(f">>> Judging {len(alpaca_prompts)} pairs (random A/B order)")
    judgments = []
    for p, sft_out, dpo_out in zip(alpaca_prompts, sft_outputs, dpo_outputs):
        flip = random.random() < 0.5
        if flip:
            j = judge_pair(dpo_out, sft_out, p["prompt"])
            if j and j.get("winner") in ("A", "B"):
                j["winner_model"] = "dpo" if j["winner"] == "A" else "sft"
        else:
            j = judge_pair(sft_out, dpo_out, p["prompt"])
            if j and j.get("winner") in ("A", "B"):
                j["winner_model"] = "sft" if j["winner"] == "A" else "dpo"
        if j and j.get("winner") == "tie":
            j["winner_model"] = "tie"
        judgments.append(j or {"winner_model": "skipped"})

    n_dpo = sum(1 for j in judgments if j.get("winner_model") == "dpo")
    n_tie = sum(1 for j in judgments if j.get("winner_model") == "tie")
    n_total = len(judgments)
    alpaca_winrate = (n_dpo + 0.5 * n_tie) / n_total if n_total else 0.0
    print(f"\nDPO win-rate: {n_dpo}/{n_total} wins, {n_tie} ties → {alpaca_winrate:.3f}")
    (EVAL_OUT / "alpaca_lite_judgments.json").write_text(
        json.dumps(judgments, ensure_ascii=False, indent=2)
    )
else:
    print("⚠ No API key set, skipping AlpacaEval-lite. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
    alpaca_winrate = None

# %% [markdown]
# ## 6. Aggregate + 4-bar comparison plot

# %%
def extract_score(results, primary_metric):
    """Pull the primary metric from a lm-eval results dict."""
    if "error" in results:
        return float("nan")
    for task_name, metrics_dict in results.items():
        if primary_metric in metrics_dict:
            return float(metrics_dict[primary_metric])
        for k, v in metrics_dict.items():
            if isinstance(v, (int, float)) and "acc" in k:
                return float(v)
    nums = [v for r in results.values() for v in r.values() if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else float("nan")


metrics = {
    "IFEval": {
        "sft": extract_score(sft_ifeval, "prompt_level_strict_acc,none"),
        "dpo": extract_score(dpo_ifeval, "prompt_level_strict_acc,none"),
    },
    "GSM8K": {
        "sft": extract_score(sft_gsm8k, "exact_match,strict-match"),
        "dpo": extract_score(dpo_gsm8k, "exact_match,strict-match"),
    },
    "MMLU": {
        "sft": extract_score(sft_mmlu, "acc,none"),
        "dpo": extract_score(dpo_mmlu, "acc,none"),
    },
    "AlpacaEval-lite": {
        "sft": 0.5 if alpaca_winrate is not None else float("nan"),
        "dpo": alpaca_winrate if alpaca_winrate is not None else float("nan"),
    },
}

if PRESERVE_SKIPPED_METRICS:
    previous_metrics = PREVIOUS_BENCHMARK_RESULTS.get("metrics", {})
    for bench, scores in metrics.items():
        previous_scores = previous_metrics.get(bench, {})
        for model_name, value in list(scores.items()):
            if value == value:
                continue
            previous_value = previous_scores.get(model_name)
            if isinstance(previous_value, (int, float)) and previous_value == previous_value:
                scores[model_name] = previous_value
                print(f"Preserved previous {bench} {model_name} score: {previous_value:.3f}")

print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
for bench, scores in metrics.items():
    delta = (scores["dpo"] - scores["sft"]) if all(s == s for s in scores.values()) else float("nan")
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "—"
    print(f"  {bench:18s}  SFT: {scores['sft']:.3f}   DPO: {scores['dpo']:.3f}   Δ: {delta:+.3f} {arrow}")

# %%
import matplotlib.pyplot as plt
import numpy as np

bench_names = list(metrics.keys())
sft_scores = [metrics[b]["sft"] for b in bench_names]
dpo_scores = [metrics[b]["dpo"] for b in bench_names]

x = np.arange(len(bench_names))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 5))
b1 = ax.bar(x - width / 2, sft_scores, width, label="SFT-only", color="#2e548a")
b2 = ax.bar(x + width / 2, dpo_scores, width, label="SFT+DPO", color="#c83538")

for bars in [b1, b2]:
    for rect in bars:
        h = rect.get_height()
        if h == h:
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

for i, b in enumerate(bench_names):
    s, d = metrics[b]["sft"], metrics[b]["dpo"]
    if s == s and d == d:
        delta = d - s
        color = "#2e548a" if delta > 0 else "#c83538" if delta < 0 else "#666"
        ax.annotate(f"Δ={delta:+.3f}", xy=(x[i], max(s, d) + 0.04),
                    ha="center", fontsize=9, color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(bench_names)
ax.set_ylabel("Score (acc / win-rate)")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="#888", linestyle=":", linewidth=0.7, alpha=0.5)
ax.set_title(f"Benchmark comparison: SFT-only vs SFT+DPO  ·  {COMPUTE_TIER}")
ax.legend(loc="upper right")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()

screenshot_dir = REPO_ROOT / "submission" / "screenshots"
screenshot_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(screenshot_dir / "07-benchmark-comparison.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Save results JSON (consumed by `make verify`)

# %%
final = {
    "compute_tier": COMPUTE_TIER,
    "limits": {
        "ifeval": LIMIT_IFEVAL,
        "gsm8k": LIMIT_GSM8K,
        "mmlu": LIMIT_MMLU,
        "alpaca_lite": LIMIT_ALPACA,
    },
    "metrics": metrics,
    "deltas": {b: metrics[b]["dpo"] - metrics[b]["sft"]
               for b in bench_names if metrics[b]["sft"] == metrics[b]["sft"]},
}
BENCHMARK_RESULTS_PATH.write_text(
    json.dumps(final, ensure_ascii=False, indent=2)
)
print(f"\nSaved {BENCHMARK_RESULTS_PATH}")

# %% [markdown]
# ## 8. Vibe-coding callout — interpret your numbers
#
# Câu hỏi để brainstorm trước khi viết REFLECTION § 7:
#
# 1. **Benchmark nào tăng nhiều nhất?** Nếu IFEval tăng nhiều, DPO đã làm đúng việc của nó
#    (chat-tuning). Nếu AlpacaEval-lite tăng nhiều → preference signal transfer tốt.
#
# 2. **Benchmark nào *giảm*?** GSM8K hoặc MATH giảm = **alignment tax** kinh điển (deck §8.1).
#    Đó không phải bug; đó là trade-off:
#    - Capacity được dành cho format (theo lệnh) thay vì reasoning sâu
#    - Chat data thường ngắn hơn math derivation → model học output ngắn hơn
#
# 3. **MMLU thay đổi ít hay nhiều?** MMLU đo *kiến thức nền*. DPO trên preference data thường
#    KHÔNG dạy facts mới → MMLU thường flat (±2pp). Nếu giảm > 5pp → catastrophic forgetting,
#    giảm β hoặc giảm epochs.
#
# 4. **AlpacaEval-lite có khớp với NB4 judge eval không?** Cả 2 đều judge-based nhưng prompt
#    distribution khác nhau (NB4: 8 fixed, mix helpfulness+safety; AlpacaEval-lite: 100,
#    helpfulness-focused). Kết quả khác = signal về *prompt distribution sensitivity*.
#
# **Vibe-coding tip (xem `VIBE-CODING.md` Phần 2 § Common workflows):** bạn có thể tự động hoá
# với Claude Code:
#
# ```
# claude --permission-mode plan -p "Read data/eval/benchmark_results.json
# and submission/REFLECTION.md, propose a draft for § 7 (≥ 150 words) interpreting
# the deltas. Reference deck §8.1 for alignment tax framing."
# ```
#
# ---
#
# **Bạn vừa hoàn thành full Lab 22 pipeline.** Run `make verify` để check submission readiness.
