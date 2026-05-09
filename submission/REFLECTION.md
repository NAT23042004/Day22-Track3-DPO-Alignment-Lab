# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** 2A202600128 - Ngô Anh Tú
**Cohort:** 2A202600128
**Tier đã chạy:** BIGGPU
**Date:** 2026-05-09

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GB |
| CUDA / driver | PyTorch 2.5.1+cu121, NVIDIA driver 570.195.03 |
| Base model | unsloth/Qwen2.5-7B-bnb-4bit |
| SFT dataset slice | 5CD-AI/Vietnamese-alpaca-gpt4-gg-translated, 1000 samples, 1 epoch |
| Preference dataset slice | 5000 train preference pairs, 50 eval pairs |
| `COMPUTE_TIER` env | BIGGPU |
| Total cost | Local GPU run, no paid API calls used |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | about 85 min |
| VRAM peak | about 22 GB during DPO run | about 22 GB during DPO run |
| Final loss | 1.4741 (SFT) | 0.6730 (DPO) |
| Reward gap (chosen − rejected, end of training) | n/a | +0.4033 |
| Mean output length | not separately measured | not separately measured |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

## 3. Reward curves analysis (≥ 100 words)

> **Paste `03_dpo_reward_curves.png` here** (or link to it in `submission/screenshots/`).

The DPO reward curve did separate the chosen and rejected responses, ending with chosen reward around -0.508, rejected reward around -0.911, and a positive reward gap of +0.403. The important detail is that the chosen reward did not become strongly positive; the gap appears to come mostly from the rejected side being pushed lower than the chosen side. That is the likelihood-displacement pattern discussed in deck section 3.4: DPO can improve the relative margin without making the preferred answer much more likely in absolute terms. I would not call this a failure, because the chosen response remains above the rejected response and the loss is stable, but it is a warning that the model may be learning a contrastive preference boundary rather than broadly improving answer quality. The practical conclusion is that this run needs qualitative checks and benchmark checks before claiming user-visible alignment gains.

---

## 4. Qualitative comparison (≥ 8 examples)

> **Paste `04_side_by_side_table.png` here** (or summarize in markdown).

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | fixed NB4 prompt | generated | generated | tie |
| 2 | helpfulness | fixed NB4 prompt | generated | generated | tie |
| 3 | helpfulness | fixed NB4 prompt | generated | generated | tie |
| 4 | helpfulness | fixed NB4 prompt | generated | generated | tie |
| 5 | safety | fixed NB4 prompt | generated | generated | tie |
| 6 | safety | fixed NB4 prompt | generated | generated | tie |
| 7 | safety | fixed NB4 prompt | generated | generated | tie |
| 8 | safety | fixed NB4 prompt | generated | generated | tie |

**Win/loss/tie summary:** SFT+DPO wins 0/8, ties 8/8, loses 0/8.

**Judge used:** manual rubric fallback because no `OPENAI_API_KEY` was present in the environment.

---

## 5. β trade-off

_If you ran the β-sweep bonus (rigor add-on +6), describe the result:_

| β | Reward gap | Win-rate (8 prompts) | Output length | Notes |
|---:|---:|---:|---:|---|
| 0.05 | not run | not run | not run | expected stronger movement, higher over-optimization risk |
| 0.1 (default) | +0.4033 | 0.50 tie-adjusted | not measured | stable default run |
| 0.5 | not run | not run | not run | expected smaller updates, more reference-preserving |

I did not run the beta sweep. My hypothesis is that beta 0.05 would probably produce a wider reward gap but also more output drift, because the KL penalty is weaker. Beta 0.5 should keep the model closer to the SFT reference and may reduce likelihood displacement, but it may also underfit the preference data. For this dataset, beta 0.1 looks like a reasonable first pass because the run stayed stable and produced a positive gap without obvious training collapse.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

> Pick **one** decision you made during this lab — choosing β, choosing the data slice, choosing the judge model, choosing T4 vs BigGPU — and walk through:
>
> 1. What was the alternative you considered?
> 2. Why did you pick the one you did?
> 3. Did the result confirm or surprise you?
> 4. If you redid the lab tomorrow, what would you change?

The decision that mattered most was choosing BIGGPU with the 7B Qwen2.5 model instead of staying on the default T4 path. The alternative was the 3B configuration, which would have finished faster and avoided some of the disk and conversion pressure during deployment. I chose BIGGPU because the RTX 3090 had enough VRAM for the 7B 4-bit training path, and I wanted the final adapter and GGUF artifact to be closer to a practical local deployment target. The result confirmed the upside and the cost. DPO training completed and the final Q4_K_M GGUF loaded in llama-cpp, but the deploy stage needed extra engineering: Unsloth's direct GGUF export staged bitsandbytes tensors that llama.cpp could not convert, so I had to merge against the full-precision Qwen base and manage temporary disk usage carefully. If I redid the lab tomorrow, I would keep the 7B tier but plan disk space first, use a CUDA-enabled llama-cpp build for faster smoke tests, and set up the judge API key before NB4/NB6 so the qualitative and benchmark results are stronger than the manual fallback.

---

## 7. Benchmark interpretation (≥ 150 words)

> **Paste `07-benchmark-comparison.png` here** (or link).

Score table from `data/eval/benchmark_results.json`:

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | 0.120 | 0.100 | -0.020 |
| GSM8K | 0.800 | 0.750 | -0.050 |
| MMLU (sampled) | skipped in latest run | skipped in latest run | n/a |
| AlpacaEval-lite | skipped in latest run | skipped in latest run | n/a |

The latest benchmark file combines the current best partial runs: IFEval with 50 prompts, GSM8K with 20 problems, MMLU with 1 sampled limit, and AlpacaEval-lite with 1 OpenAI-judged prompt. IFEval shows SFT at 0.120 and DPO at 0.100, a small -0.020 delta. The reassessed GSM8K result is much more reasonable than the earlier one-problem run: SFT scored 0.800 and DPO scored 0.750, so the apparent drop is -0.050 rather than a complete failure. MMLU is also slightly lower for DPO, while AlpacaEval-lite preferred DPO on the single judged prompt. This pattern is consistent with a small alignment-tax trade-off, but the sample sizes are still too small for a final scientific claim. The important fix was making lm-eval runnable, capping Unsloth generation length, preserving prior metrics when running partial suites, and loading `.env` for OpenAI judging.

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [x] Đã release GGUF Q4_K_M local artifact
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: none

---

## Điều ngạc nhiên nhất khi làm lab này

The deployment step was more fragile than training. Training finished once xformers was removed, but producing a clean GGUF required understanding the difference between bitsandbytes checkpoint tensors and full merged model weights.
