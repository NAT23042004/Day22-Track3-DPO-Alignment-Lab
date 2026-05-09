# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB5 — Merge + Deploy + GGUF
#
# **Stack:** Unsloth `merge_and_unload` + `save_pretrained_gguf(quantization='Q4_K_M')`
# + llama-cpp-python smoke test.
# Maps to deck §7.1 lab brief: "merge adapter, quantize GGUF, serve với vLLM".
#
# > **Mục tiêu:** export the SFT+DPO adapter as a deployable GGUF Q4_K_M file
# > (~1.5 GB on 3B / ~4 GB on 7B), then smoke-test it through llama-cpp-python.
# > Final cell shows the optional vLLM serving command (BigGPU only).

# %% [markdown]
# ## 0. Setup

# %%
import os
import json
import gc
import shutil
import subprocess
import sys
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
BASE_MODEL = (
    "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4"
    else "unsloth/Qwen2.5-7B-bnb-4bit"
)
FULL_BASE_MODEL = (
    "Qwen/Qwen2.5-3B" if COMPUTE_TIER == "T4"
    else "Qwen/Qwen2.5-7B"
)
MAX_LEN = 512 if COMPUTE_TIER == "T4" else 1024

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DPO_PATH = REPO_ROOT / "adapters" / "dpo"
MERGED_PATH = REPO_ROOT / "adapters" / "merged-fp16"
GGUF_DIR = REPO_ROOT / "gguf"
MERGED_PATH.mkdir(parents=True, exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)
BF16_GGUF = GGUF_DIR / f"{FULL_BASE_MODEL.split('/')[-1]}-dpo.BF16.gguf"
Q4_GGUF = GGUF_DIR / f"{FULL_BASE_MODEL.split('/')[-1]}-dpo.Q4_K_M.gguf"

assert DPO_PATH.exists(), "NB3 must run first"

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"FULL_BASE_MODEL: {FULL_BASE_MODEL}")
print(f"DPO adapter:     {DPO_PATH}")
print(f"merged output:   {MERGED_PATH}")
print(f"GGUF output:     {GGUF_DIR}")

# %%
import torch

assert torch.cuda.is_available()

# %% [markdown]
# ## 1. Load DPO model + merge adapter

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

full_cache = REPO_ROOT / ".hf-full-cache"
model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

if BF16_GGUF.exists() or Q4_GGUF.exists():
    print("Reusing existing GGUF intermediate/final artifact; skipping HF merge.")
else:
    tokenizer = AutoTokenizer.from_pretrained(FULL_BASE_MODEL, cache_dir=str(full_cache))
    model = AutoModelForCausalLM.from_pretrained(
        FULL_BASE_MODEL,
        torch_dtype=model_dtype,
        device_map={"": "cuda:0"},
        cache_dir=str(full_cache),
        low_cpu_mem_usage=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # The DPO checkpoint contains the final LoRA weights after starting from SFT.
    # Load the post-DPO adapter directly for deployment.
    model = PeftModel.from_pretrained(model, str(DPO_PATH))
    print(f"Loaded DPO adapter from {DPO_PATH}")

    # The full base checkpoint is only needed for loading. Remove its temporary
    # cache before writing merged + intermediate GGUF files so the lab fits on
    # small disks.
    if full_cache.exists():
        shutil.rmtree(full_cache)

    model = model.merge_and_unload()
    model.save_pretrained(
        str(MERGED_PATH),
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(str(MERGED_PATH))
    print(f"Saved merged HF weights to {MERGED_PATH}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# > **Note:** The DPO adapter saved in NB3 already contains the final LoRA weights
# > after DPO training initialized from SFT-mini. Loading that adapter onto the
# > base model gives the aligned policy for export.

# %% [markdown]
# ## 2. Quantize to GGUF Q4_K_M
#
# Q4_K_M is the sweet spot: ~4× compression vs FP16, minimal quality loss.
# We call llama.cpp directly so conversion starts from clean merged bf16 weights,
# not the original bitsandbytes checkpoint.

# %%
# Save GGUF in 1 quantization tier (Q4_K_M). Add more tiers below if you want the
# +3 "GGUF release published" rigor add-on.
converter = Path("/root/.unsloth/llama.cpp/unsloth_convert_hf_to_gguf.py")
quantizer = Path("/root/.unsloth/llama.cpp/llama-quantize")
assert converter.exists(), f"llama.cpp converter missing: {converter}"
assert quantizer.exists(), f"llama.cpp quantizer missing: {quantizer}"

bf16_gguf = BF16_GGUF
q4_gguf = Q4_GGUF

if not bf16_gguf.exists() and not q4_gguf.exists():
    subprocess.run(
        [
            sys.executable,
            str(converter),
            "--outfile",
            str(bf16_gguf),
            "--outtype",
            "bf16",
            "--split-max-size",
            "50G",
            str(MERGED_PATH),
        ],
        check=True,
    )

# The BF16 GGUF is self-contained. Drop merged HF shards before quantization so
# Q4_K_M can finish on disks with < 20 GB free.
if MERGED_PATH.exists():
    shutil.rmtree(MERGED_PATH)
MERGED_PATH.mkdir(parents=True, exist_ok=True)
if not q4_gguf.exists():
    subprocess.run([str(quantizer), str(bf16_gguf), str(q4_gguf), "Q4_K_M"], check=True)
if bf16_gguf.exists():
    bf16_gguf.unlink(missing_ok=True)
print(f"Saved GGUF Q4_K_M to {q4_gguf}")

# %% [markdown]
# ### 3a. Optional — additional quantization tiers (for the +3 rigor add-on)

# %%
# Uncomment if you want Q5_K_M + Q8_0 too (~2× total disk space).
# Each adds ~30s for an extra GGUF file.
#
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q5_k_m")
# model.save_pretrained_gguf(str(GGUF_DIR), tokenizer, quantization_method="q8_0")

# %%
import os

print("GGUF files:")
for p in sorted(GGUF_DIR.iterdir()):
    if p.suffix == ".gguf":
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:50s}  {size_mb:>8.1f} MB")

gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## 4. Smoke test with llama-cpp-python

# %%
from llama_cpp import Llama

# Find the Q4_K_M GGUF
gguf_files = list(GGUF_DIR.glob("*Q4_K_M*.gguf")) + list(GGUF_DIR.glob("*q4_k_m*.gguf"))
assert gguf_files, "No Q4_K_M GGUF found — step 3 may have failed"
gguf_path = gguf_files[0]
print(f"Loading: {gguf_path.name}")

# n_gpu_layers=-1 offloads all layers to GPU if compiled with CUDA/Metal/Vulkan
llm = Llama(
    model_path=str(gguf_path),
    n_ctx=MAX_LEN,
    n_gpu_layers=-1,           # all layers on GPU; falls back to CPU if no GPU compile
    verbose=False,
)
print("Loaded.")

# %% [markdown]
# ### 4a. Smoke prompt + response (deliverable: `06-gguf-smoke.png`)

# %%
SMOKE_PROMPT = "Giải thích ngắn gọn (3 câu) cách thuật toán Bubble sort hoạt động."

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": SMOKE_PROMPT}],
    max_tokens=200,
    temperature=0.0,
)

print(f"PROMPT:\n  {SMOKE_PROMPT}\n")
print(f"RESPONSE (Q4_K_M GGUF, llama-cpp-python):\n  {response['choices'][0]['message']['content']}")
print(f"\nTokens used: {response['usage']}")

# %% [markdown]
# ## 5. Optional — vLLM serving (BigGPU only)
#
# vLLM provides production-grade OpenAI-compatible serving. **Requires CUDA GPU
# with ≥ 16 GB VRAM** and `vllm` installed (see `requirements-biggpu.txt`).
# On T4 tier this cell will OOM. Skip on T4.
#
# Run in a SEPARATE terminal (NOT in the notebook — vLLM blocks until killed):
#
# ```bash
# pip install vllm                         # once
# vllm serve adapters/merged-fp16 \
#   --port 8000 \
#   --max-model-len 1024 \
#   --gpu-memory-utilization 0.9
# ```
#
# Then test:
#
# ```bash
# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model": "merged-fp16", "messages": [{"role": "user", "content": "Hello"}]}'
# ```
#
# **Why not in the notebook?** vLLM's process model doesn't play nicely with
# Jupyter — it expects to own the GPU + a long-running HTTP server. Run it as
# a sidecar process. The deck mentions vLLM as the deploy target; for actual
# production you'd containerize this command. For the lab, llama-cpp-python in
# step 4 is the graded artifact.

# %% [markdown]
# ## 6. Save deployment metadata

# %%
deploy_meta = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "merged_path": str(MERGED_PATH),
    "gguf_path": str(gguf_path),
    "gguf_size_mb": round(gguf_path.stat().st_size / 1e6, 1),
    "quantization": "q4_k_m",
    "smoke_prompt": SMOKE_PROMPT,
    "smoke_response": response["choices"][0]["message"]["content"],
}
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").parent.mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").write_text(
    json.dumps(deploy_meta, ensure_ascii=False, indent=2)
)
print("Saved data/eval/deploy_meta.json")

# %% [markdown]
# ## 7. Submission checklist
#
# Bạn vừa hoàn thành core lab. Trước khi submit:
#
# 1. **Run** `make verify` — gatekeeper sẽ list missing artifacts.
# 2. **Take screenshots** vào `submission/screenshots/` (xem `submission/screenshots/README.md`).
# 3. **Fill** `submission/REFLECTION.md` — đặc biệt là § 3 (reward curves analysis,
#    cross-reference deck §3.4) và § 6 (single change that mattered most).
# 4. **(Optional)** Pick a rigor add-on từ rubric.md (β-sweep, HF push, GGUF
#    release, W&B link, cross-judge).
# 5. **(Optional)** Pick a `BONUS-CHALLENGE.md` provocation cho creative bonus.
#
# Push public repo + paste URL vào VinUni LMS Day-22 box.
#
# Câu hỏi cuối để brainstorm trước khi đóng laptop:
#
# > **The deck says:** "DPO + 30 min A100 + 2k UltraFeedback → 3.2 → 4.1 helpfulness."
# > **You measured:** _<your win-rate from NB4>_.
# > **Why might they differ?** Dataset (English vs VN), base model (Qwen2.5-3B vs
# > deck's unspecified base), judge bias, sample size (8 prompts vs deck's full eval).
# > Đó chính là § 6 trong REFLECTION — what 1 change would close the gap.
