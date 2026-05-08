#!/usr/bin/env python3
"""CLI wrapper for NB2 logic — prepares preference Parquet.

Usage:
    python scripts/prepare_preference_data.py
    python scripts/prepare_preference_data.py --slice 5000 --output data/pref-5k

Mirrors `notebooks/02_preference_data.py` cells 1-3. Use this if you want to
re-build the data without launching Jupyter (e.g., from a Makefile target).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=os.environ.get(
            "PREF_DATASET", "argilla/ultrafeedback-binarized-preferences-cleaned"
        ),
    )
    parser.add_argument("--slice", type=int, default=None,
                        help="Number of pairs (default: 2000 for T4, 5000 for BIGGPU)")
    parser.add_argument("--output", default=str(REPO / "data" / "pref"),
                        help="Output directory")
    parser.add_argument("--tokenizer", default=str(REPO / "adapters" / "sft-mini"),
                        help="Tokenizer path (defaults to SFT-mini adapter dir)")
    args = parser.parse_args()

    tier = os.environ.get("COMPUTE_TIER", "T4").upper()
    pref_slice = args.slice or (2000 if tier == "T4" else 5000)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset}[:{pref_slice}]...")
    ds = load_dataset(args.dataset, split=f"train[:{pref_slice}]")
    print(f"  {len(ds)} pairs · cols: {ds.column_names}")

    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def render_chat(messages, *, add_generation_prompt=False):
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        text = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def fmt(row):
        prompt_msgs = [{"role": "user", "content": row["prompt"]}]
        prompt_text = render_chat(prompt_msgs, add_generation_prompt=True)
        chosen = row["chosen"][-1]["content"] if isinstance(row["chosen"], list) else row["chosen"]
        rejected = row["rejected"][-1]["content"] if isinstance(row["rejected"], list) else row["rejected"]
        return {"prompt": prompt_text, "chosen": chosen, "rejected": rejected}

    pref = ds.map(fmt, remove_columns=ds.column_names)
    pref.to_parquet(str(out / "train.parquet"))

    # 50 eval slice
    eval_slice = pref.select(range(max(0, len(pref) - 50), len(pref)))
    eval_slice.to_parquet(str(out / "eval.parquet"))

    print(f"Wrote {len(pref)} train + {len(eval_slice)} eval pairs to {out}/")


if __name__ == "__main__":
    main()
