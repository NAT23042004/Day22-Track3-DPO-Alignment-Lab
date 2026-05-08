# Day 22 — DPO/ORPO Alignment Lab (Track 3)

Lab cho **AICB-P2T3 · Ngày 22 · DPO/ORPO Alignment — From SFT to Preference Learning**.
Build SFT-mini checkpoint → train DPO adapter → compare SFT-only vs SFT+DPO → merge + GGUF + serve.

> Lab 22 là **lab alignment đầu tiên trong khoá** — bạn đi từ SFT (Lab 21) sang preference learning, đo helpfulness/safety bằng judge, và export model deployable. Output có thể là 1 **DPO-aligned VN model open-source publishable đầu tiên end-to-end của khoá** (xem deck §5).

---

## Hai tier — chọn cái phù hợp

| Tier | Compute | Base model | SFT slice | DPO slice | Time | Khi nào dùng |
|---|---|---|---|---|---|---|
| **T4 (default)** | Free Colab T4 16 GB / laptop GPU ≥ 12 GB | `Qwen2.5-3B-bnb-4bit` | 1k VN Alpaca | 2k UltraFeedback | ~75 min (incl. NB6 benchmark) | Hầu hết học viên — không Anthropic/OpenAI key, free Colab, RTX 3060/3070/4060 laptop |
| **BigGPU (full)** | Colab Pro A100/L4 / Kaggle T4×2 / cloud H100 | `Qwen2.5-7B-bnb-4bit` | 1k VN Alpaca | 5k UltraFeedback | ~60 min (incl. NB6 benchmark) | Đã có cloud GPU, muốn faithful với deck demo (3.2 → 4.1 helpfulness, A100 timing) |

> Cả hai tier dùng **cùng notebook source** — đổi giữa T4 và BigGPU bằng cách sửa `COMPUTE_TIER` trong `.env` (hoặc đổi badge launch URL bên dưới).

> **VRAM math quan trọng:** DPO cần *cả policy và reference* model trong memory cùng lúc → ~2× SFT VRAM. Một 7B QLoRA SFT vừa 10 GB nhưng 7B QLoRA DPO cần ~18-20 GB. Đó là lý do T4 tier dùng 3B (không 7B) và BigGPU tier yêu cầu A100/L4.

---

## Quick Start — T4 (recommended)

**Option 1: Free Colab (zero install)**

[![Open T4 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/Day22-Track3-DPO-Alignment-Lab/blob/main/colab/Lab22_DPO_T4.ipynb)

Click → Runtime → Change runtime type → **T4 GPU** → Run all.

**Option 2: Local laptop (≥ 12 GB VRAM)**

```bash
git clone https://github.com/<your-username>/Day22-Track3-DPO-Alignment-Lab.git
cd Day22-Track3-DPO-Alignment-Lab
bash setup-laptop.sh    # ~5 min — venv + deps + cuda probe + smoke test
make smoke              # 2-step training run on each notebook to verify GPU
make pipeline           # full pipeline: sft → data → dpo → eval → deploy (~45 min)
make verify             # pre-submission gatekeeper
```

Yêu cầu: **Python 3.10–3.12**, NVIDIA GPU ≥ 12 GB VRAM (3060/4060 trở lên), CUDA 11.8 hoặc 12.1+.

### Tất cả lệnh `make`

```
make help            Show this help
make setup           Auto-detect Colab vs laptop, install deps + smoke check
make smoke           2-step training run on each notebook to verify imports + GPU
make sft             NB1 — build SFT-mini checkpoint (~10 min T4 / ~5 min A100)
make data            NB2 — preference data prep (~2 min)
make dpo             NB3 — full DPO training (~30 min T4 / ~20 min A100)
make eval            NB4 — side-by-side comparison + optional API judge
make deploy          NB5 — merge + GGUF + llama.cpp smoke
make bench           NB6 — IFEval/GSM8K/MMLU/AlpacaEval-lite + 4-bar plot (~30 min T4)
make pipeline        Run NB1 → NB6 in order
make beta-sweep      Bonus rigor: re-run NB3 with β ∈ {0.05, 0.1, 0.5}
make verify          scripts/verify.py — pre-submission gatekeeper
make clean           rm adapters/ data/pref/ gguf/ __pycache__
```

---

## Quick Start — BigGPU (full)

```bash
bash setup-laptop.sh                     # base install
pip install -r requirements-biggpu.txt   # adds vllm, flash-attn, deepspeed
echo 'COMPUTE_TIER=BIGGPU' > .env        # flip the tier flag
make pipeline                             # ~30 min on A100
```

Hoặc Colab Pro / Kaggle: open `colab/Lab22_DPO_BigGPU.ipynb` (badge link sẽ resolve sau khi push lên GitHub).

---

## Cấu trúc & tiến trình

| Notebook | Skill | Slide deliverable | Pass when… |
|---|---|---|---|
| `01_sft_mini` | Re-build Lab 21 SFT checkpoint inline (Unsloth + LoRA r=16, 1k VN Alpaca, 1 epoch) | Bullet 1 — base SFT artifact | adapter saves; loss decreases monotonically |
| `02_preference_data` | Load `argilla/ultrafeedback-binarized-preferences-cleaned`, format `prompt/chosen/rejected`, save Parquet | Bullet 2 — preference data ready | parquet written; chosen ≠ rejected; 3 examples printed |
| `03_dpo_train` | TRL `DPOTrainer(beta=0.1, lr=5e-7)` on SFT model + frozen reference; plot reward curves | Bullet 3 — DPO training + reward curves | adapter saves; reward gap > 0; chosen reward ↑ (or ↓ explained per deck §3.4) |
| `04_compare_and_eval` | 8 fixed prompts × {SFT, SFT+DPO} side-by-side; optional GPT-4o/Claude judge | Bullet 4 — helpfulness comparison | table renders; ≥ 8 examples; win/loss/tie counts reported |
| `05_merge_deploy_gguf` | `merge_and_unload()` → GGUF Q4_K_M → llama-cpp-python smoke test | Bullet 5 — deployable artifact | GGUF < 5 GB; smoke prompt returns coherent VN |
| `06_benchmark` | IFEval + GSM8K + MMLU (sampled) + AlpacaEval-lite on SFT-only vs SFT+DPO; 4-bar comparison plot | Bullet 6 — quantitative benchmark | `benchmark_results.json` written; 4 deltas annotated in plot; Reflection §7 explains alignment-tax pattern |

**Source format:** Notebooks live as Jupytext `.py` files (small, easy to review). `setup-laptop.sh` and `make smoke` auto-convert to `.ipynb`. Edit `.ipynb` in Jupyter and Jupytext keeps both in sync.

**Colab variant:** `colab/Lab22_DPO_T4.ipynb` and `colab/Lab22_DPO_BigGPU.ipynb` are stitched-together single-file `.ipynb` — same content as the 5 Jupytext sources but ready to launch via badge. Pick the laptop path or the Colab path; both produce identical artifacts.

---

## Slide section → notebook map

Tra ngược từ slide bạn nhớ trong lecture về cell trong notebook:

| Deck section | Slide topic | Notebook |
|---|---|---|
| §1 (Tại sao SFT chưa đủ?) | Distribution shift, KL drift | `01_sft_mini.py` (mục đích) |
| §3.1 (DPO loss derivation) | Bradley-Terry → log-ratio | `03_dpo_train.py` cell §3 |
| §3.2 (β tuning) | Trade-off conservative vs aggressive | `03_dpo_train.py` cell §5 (bonus β-sweep) |
| §3.4 (Failure modes) | Likelihood displacement, length hacking | `03_dpo_train.py` warning cell |
| §5.2 (TRL implementation) | `DPOConfig` hyperparameters | `03_dpo_train.py` cell §2 |
| §5.4 (VN landscape) | VinaLLaMA / PhoGPT / Vistral / SeaLLM | `02_preference_data.py` callout + `BONUS-CHALLENGE.md` provocation 1 |
| §8.1–§8.5 (Đánh giá Alignment) | Static / Judge / Reward-Model / VN landscape | `06_benchmark.py` |
| §9.1 (Demo) | UltraFeedback 2k, 30 min A100, 3.2 → 4.1 | `04_compare_and_eval.py` |
| §9.2b (Tulu 3 stats) | +1.7 MATH / +3.3 GSM8K / +1.3 IFEval | reference numbers + `06_benchmark.py` measures *your* equivalents |

---

## Deliverable (6 notebook đã chạy + ảnh chụp + reflection)

Mapping 1-to-1 với slide deliverable bullets:

1. **NB1** — `adapters/sft-mini/` written; `01_sft_loss.png` shows monotonic decrease.
2. **NB2** — `data/pref/train.parquet` with prompt/chosen/rejected columns; 3 inspected examples printed.
3. **NB3** — `adapters/dpo/` written; reward gap plot saved as `03_dpo_reward_curves.png`.
4. **NB4** — `04_side_by_side_table.png` + win/loss/tie summary (8 prompts × 2 models).
5. **NB5** — `gguf/lab22-dpo-Q4_K_M.gguf` exists; `06_gguf_smoke.png` shows llama.cpp output.
6. **NB6** — `data/eval/benchmark_results.json` + `07-benchmark-comparison.png` 4-bar chart with deltas annotated; REFLECTION §7 interprets alignment-tax pattern (deck §8.1).

Chấm điểm: xem [`rubric.md`](rubric.md). **Tổng 100 pts → Track-3 Daily Lab (30%)** + 20 pts bonus rigor add-ons (β-sweep, HF push, W&B, GGUF release).

---

## Tech stack

| Layer | Tool | Version | Why |
|---|---|---|---|
| **Training** | Unsloth | ≥ 2025.10 | Patched kernels, 7B-on-T4 viable, matches Day 21 reference |
| **Trainers** | TRL | ≥ 0.12, < 0.20 | `DPOTrainer` + `DPOConfig` (deck §5.2 surface) |
| **Adapters** | PEFT | ≥ 0.13 | LoRA r=16 α=32; reference model loaded as frozen 4-bit |
| **Quantization** | bitsandbytes | ≥ 0.44 | NF4 base + bf16 LoRA |
| **Data** | datasets + pyarrow | ≥ 3.1 | UltraFeedback + VN Alpaca slices |
| **Local serving** | llama-cpp-python | ≥ 0.3 | GGUF Q4_K_M smoke test (CPU/Metal/CUDA) |
| **Cloud serving** | vllm (BigGPU only) | ≥ 0.6.4 | OpenAI-compat for production-style serve test |
| **Plotting** | matplotlib + pandas | ≥ 3.9 | Reward curves + side-by-side tables |

**Why not vLLM by default?** vLLM needs CUDA GPU + ≥ 16 GB VRAM and adds 3-5 min Docker/CUDA-toolkit install. For T4 tier we use llama-cpp-python which compiles inline in the wheel and works on CPU/Metal/CUDA. BigGPU tier gets vLLM as a final cell (informational on T4).

---

## Vibe-coding tips

Lab này thiết kế cho **vibe-coding era**: bạn dùng AI assistant trong terminal (Claude Code, Codex CLI, OpenCode) để generate boilerplate, focus vào *judgment decisions* — chọn dataset, chọn β, đọc reward curve, judge output. Đọc [`VIBE-CODING.md`](VIBE-CODING.md) **trước khi bắt đầu NB1** (5–10 phút) — file đó là general primer cover:

- Spec-Driven Development (SDD) và TDD trong LLM era
- Khi nào delegate cho AI, khi nào tự nghĩ
- 5 prompt patterns DPO-specific (diagnose chosen reward drop, generate VN prompts, critique config, translate UltraFeedback, judge outputs)
- CLI tool recommendations (Claude Code / Codex CLI / OpenCode)
- 3 anti-patterns phổ biến trong alignment work

Mỗi notebook cũng có **vibe-coding callout** ở cuối: nói rõ subtask nào *nên* delegate cho AI, subtask nào *phải* bạn tự nghĩ (hint: reward curve interpretation và β chọn = think-hard zone).

---

## Bonus Challenge — Build something real (optional, ungraded)

Một sân chơi **không có điểm số** — không deadline, không rubric. Mục đích: cho bạn đem **domain knowledge cá nhân** vào 1 model align thật, ship như sản phẩm cho 1 audience cụ thể. Mỗi provocation hỏi bạn 4 câu: *Ai dùng?* — *Bạn đem domain gì vào?* — *Model làm gì cho họ?* — *Output ship như thế nào?*

Đề xuất 5 provocations sẵn — bạn pick 1 hoặc invent your own:

1. **Subject tutor** cho môn bạn đang học (toán, hoá, sử, lập trình...) — scaffolded pedagogy, không phải đáp án
2. **Customer-service chatbot** cho 1 doanh nghiệp Việt cụ thể (cafe, shop, sửa xe...) — on-brand, có CTA
3. **Job-shadow assistant** cho 1 nghề bạn quan sát (Grab driver, shipper, lễ tân...) — ngắn, action-oriented
4. **Domain-safe assistant** cho 1 lĩnh vực nhạy cảm (sức khoẻ tinh thần, pháp lý, tài chính...) — có boundary rõ + hotline VN
5. **Style mimic** — model viết kiểu 1 người/tổ chức bạn admire

Full provocations: [`BONUS-CHALLENGE.md`](BONUS-CHALLENGE.md) (tiếng Việt) · [`BONUS-CHALLENGE-EN.md`](BONUS-CHALLENGE-EN.md) (English). Format: brainstorm-first, code-second, làm đôi/triple OK. Output: 1 portfolio piece có thể chỉ vào nói "tôi build cái này, audience X, dùng để Y."

> Bonus **không** ảnh hưởng core grade. Phần thưởng thực sự là 1 portfolio piece phục vụ *ai đó cụ thể* + feedback bằng văn bản từ giảng viên về *application thinking* của bạn.

---

## Cấu trúc repo

```
.
├── README.md                       # bạn đang đọc
├── HARDWARE-GUIDE.md               # T4 vs BigGPU decision tree
├── VIBE-CODING.md                  # vibe-coding workflow tips (5-10 phút đọc)
├── BONUS-CHALLENGE.md              # creative sandbox brief (tiếng Việt)
├── BONUS-CHALLENGE-EN.md           # creative sandbox brief (English)
├── rubric.md                       # 100-pt grading + 20 pt bonus rigor add-ons
├── Makefile                        # tier-aware orchestration
├── setup-colab.sh                  # one-line Colab install
├── setup-laptop.sh                 # local venv + cuda probe
├── requirements.txt                # T4 baseline deps
├── requirements-biggpu.txt         # BigGPU extras (vllm, flash-attn)
├── pyproject.toml                  # for `uv` users
├── .env.example                    # env template (COMPUTE_TIER, API keys)
├── notebooks/                      # 6 Jupytext .py files (source of truth)
│   ├── 01_sft_mini.py              # build SFT checkpoint inline
│   ├── 02_preference_data.py       # load + format UltraFeedback
│   ├── 03_dpo_train.py             # TRL DPOTrainer + reward curves
│   ├── 04_compare_and_eval.py      # SFT-only vs SFT+DPO + judge
│   ├── 05_merge_deploy_gguf.py     # merge + GGUF + llama.cpp smoke
│   └── 06_benchmark.py             # IFEval/GSM8K/MMLU/AlpacaEval-lite + 4-bar plot
├── colab/                          # Colab-launchable .ipynb mirrors
│   ├── Lab22_DPO_T4.ipynb
│   └── Lab22_DPO_BigGPU.ipynb
├── scripts/
│   ├── prepare_preference_data.py  # CLI wrapper for NB2 logic
│   ├── train_dpo.py                # CLI wrapper for NB3 logic
│   ├── eval_judge.py               # OpenAI/Anthropic judge — falls back to manual
│   ├── merge_and_gguf.py           # CLI wrapper for NB5 logic
│   └── verify.py                   # pre-submission gatekeeper
├── data/                           # gitignored; populated by NB2 / scripts
├── adapters/                       # gitignored; SFT + DPO outputs
├── submission/
│   ├── REFLECTION.md               # personal report template (6 sections)
│   └── screenshots/                # add 6 required + 3 optional screenshots
└── solutions/                      # released after submission deadline
    └── README.md
```

---

## Common gotchas

| Triệu chứng | Fix |
|---|---|
| OOM ngay khi load model | Đang dùng tier sai. T4 → `Qwen2.5-3B`; nếu vẫn OOM, restart runtime + downgrade `unsloth` 1 minor |
| `chosen_rewards` không tăng | Bình thường ở 100 step đầu. Sau 500 step nếu vẫn flat → giảm `beta` 0.1 → 0.05 hoặc tăng `lr` 5e-7 → 1e-6 |
| `chosen_rewards` *giảm* mà reward gap *tăng* | Đó là **likelihood displacement** (deck §3.4). Bình thường ở DPO; ghi vào REFLECTION § "β trade-off" |
| `RuntimeError: padding token is not set` | Add `tokenizer.pad_token = tokenizer.eos_token` trước khi tạo trainer |
| Unsloth + TRL version mismatch | Pin: `unsloth>=2025.10 trl>=0.12,<0.20`. Nếu lỗi sau Unsloth update, downgrade Unsloth |
| GGUF merge fails với "tied weights" | Xoá `model.config.tie_word_embeddings` trước `merge_and_unload()` |
| Colab T4 OOM at DPO step 1 | Tăng `gradient_accumulation_steps` 8 → 16, giảm `per_device_train_batch_size` 1 → 1 (already min), giảm `max_length` 512 → 384 |
| llama-cpp-python wheel install fails | `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python` (CUDA), `-DGGML_METAL=on` (Mac) |
| `lm_eval` import fails | `pip install "lm-eval[ifeval,math]>=0.4.5"` — extras pull `langdetect` and `sympy` |
| NB6 IFEval crashes with "DataLoader" worker | lm-eval 0.4.x compat — set env `HF_DATASETS_TRUST_REMOTE_CODE=1` and rerun |
| NB6 GSM8K accuracy = 0.000 | Few-shot prompts not loading. Verify `--num_fewshot 8` reaches the harness; downgrade lm-eval if 0.4.6 ships changes |
| NB6 takes > 90 min on T4 | Lower `LIMIT_MMLU` (default 500) and `LIMIT_GSM8K` (default 500) further. Bench tier checks env first. |

---

## Submission

**KHÔNG cần PR — chỉ submit GitHub URL công khai vào VinUni LMS.**

1. **Fork hoặc copy repo này lên GitHub account của bạn**, set repo **public**.
   ```bash
   git init -b main
   git remote add origin https://github.com/<your-username>/Day22-Track3-DPO-Alignment-Lab.git
   ```
2. Hoàn thành 5 notebooks (giữ output cells trong `.ipynb`).
3. Add ảnh chụp vào `submission/screenshots/` (xem [`submission/screenshots/README.md`](submission/screenshots/README.md) để biết list 6+3).
4. Điền [`submission/REFLECTION.md`](submission/REFLECTION.md) (6 sections, ≥150 từ §3 + §6).
5. `make verify` — pre-submission gatekeeper. Nếu fail, fix và rerun.
6. Push lên public repo:
   ```bash
   git add -A
   git commit -m "Lab 22 submission — <Họ Tên>"
   git push -u origin main
   ```
7. **Paste public GitHub URL của bạn vào ô submission của Day 22 trong VinUni LMS.** Không cần PR. Không cần fork-back.

> **Quan trọng:** Repo phải **public** đến khi điểm được công bố. Nếu private, grader không xem được → 0 điểm.

**Submission Options A / B / C** (cùng convention với Day 21):
- **A — Lightweight ZIP** (default): GitHub repo + executed notebooks + screenshots + REFLECTION
- **B — Professional** (+5 bonus): A + adapters pushed to HuggingFace Hub via `huggingface-cli upload`
- **C — Code-only**: Repo + report, không weights (cho học viên hết storage Colab)

---

## Acknowledgments

- **Slide deck:** [`day22/day07-dpo-orpo-alignment-tu-sft-en-preference-learning.tex`](../day07-dpo-orpo-alignment-tu-sft-en-preference-learning.tex)
- **Sibling Day 21 lab** (LoRA/QLoRA fine-tuning, the SFT predecessor): [VinUni-AI20k/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA](https://github.com/VinUni-AI20k/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA)
- **Stack:** Unsloth (Daniel Han + Mike Han), TRL (Hugging Face), PEFT, bitsandbytes, llama.cpp
- **Datasets:** UltraFeedback (Argilla), `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`

---

© VinUniversity AICB program · A20 cohort 2026 · Track 3 Day 22.
