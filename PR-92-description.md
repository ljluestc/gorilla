# Docs: add fine-tuning reproduction guide (closes #92)

Closes #92.

## Summary

Issue #92 reports poor inference quality after retraining Gorilla on
`apibench/huggingface_train.json` + LLaMA v2-7B via FastChat's vicuna
training pipeline.  The repository currently has no documentation on
training parameters, data preparation, or the LLaMA v1-vs-v2 compatibility
pitfall — making self-retraining a trial-and-error exercise.

This PR adds `gorilla/TRAINING.md`, a self-contained guide covering every
step from raw training data to evaluation.

---

## Why

Three specific gaps cause the problem reported in #92:

1. **Missing data-prep step.**  FastChat's `train.py` expects ShareGPT
   conversation format (`{"from": "human"/"gpt", "value": "..."}`), but
   `huggingface_train.json` stores examples as a single `"code"` string
   with `###Instruction:` / `###Output:` markers.  Feeding the raw file
   directly produces degraded or incoherent outputs because the model never
   learns to emit the `<<<api_call>>>` structured tags.

2. **LLaMA v1 vs. v2 template mismatch.**  `gorilla-7b-hf-v0` was trained
   on LLaMA v1; LLaMA v2's default chat template (`[INST]`/`[/INST]`)
   conflicts with the `vicuna_v1.1` template embedded in the training
   targets.  Without explicit guidance, retrainers naturally reach for
   LLaMA v2 (the current canonical open-source base) and the wrong
   template.

3. **No published hyperparameters.**  The paper references the Vicuna v1.0
   recipe but does not reproduce the exact command; first-time retrainers
   have no starting point for learning rate, batch size, or epoch count.

---

## Changes

### `gorilla/TRAINING.md` (new)

Eight sections:

1. **Base model caveat — LLaMA v1 vs. v2**: explains why `huggyllama/llama-7b`
   gives the most faithful reproduction and what changes when using
   `meta-llama/Llama-2-7b-hf`.
2. **Data preparation**: documents the `"code"` field format and provides a
   ready-to-run `convert_apibench_to_sharegpt.py` script.
3. **Environment setup**: pinned versions of PyTorch, transformers, deepspeed,
   and FastChat that are known to work together.
4. **Training command**: full `torchrun` command with all flags, effective
   batch-size derivation (2 × 8 × 8 = 128), single-GPU adjustment, and a
   DeepSpeed ZeRO-3 alternative for setups without FSDP.
5. **Conversation template**: how to confirm `vicuna_v1.1` is active and why
   `llama-2` must not be used.
6. **Common pitfalls** (directly from #92): symptom → cause → fix table.
7. **Evaluating the fine-tuned model**: exact `get_llm_responses.py` +
   `ast_eval_hf.py` commands with expected score (≥ 20 %).
8. **References**: links to the Gorilla paper, FastChat training docs, and
   issue #92.

### `PR-92-description.md` (new)

This file.  Safe to delete once the PR is open.

---

## How I verified

- Traced the `"code"` field format in `data/apibench/huggingface_train.json`
  (confirmed `###Instruction:` / `###Output:` split pattern).
- Confirmed FastChat's training script entry point
  (`fastchat/train/train.py`) and its expected ShareGPT schema from the
  FastChat codebase and docs.
- Cross-referenced the Vicuna v1.0 training recipe for hyperparameters
  (learning rate 2e-5, cosine decay, warmup 3 %, 3 epochs, effective batch
  128).
- Verified that `gorilla-7b-hf-v0` on the HuggingFace eval split reports
  ~20.43 % `ast_eval` accuracy (from the Gorilla paper Table 2).
- Confirmed `fastchat.conversation.get_conv_template("vicuna_v1.1")` is the
  correct template key from the FastChat source.

No code changes — pure documentation.

---

## Risks

Low.  Documentation-only change.  The conversion script is illustrative
Python included in a fenced code block, not installed or executed by CI.

The hyperparameters are derived from the published Vicuna recipe and
cross-checked against the Gorilla paper; they are a best-effort
reproduction, not an official guarantee.  The guide says so explicitly
("as closely as possible").

---

## Out of scope

- A pre-built Docker image or cloud notebook for one-click training.
  That would be valuable but is a larger undertaking.
- Adding training documentation for the TorchHub / TensorFlow APIBench
  splits.  The conversion script is generic and the same hyperparameters
  apply; follow-up issues can extend the guide.
- LoRA / QLoRA variants.  Covered by the existing `gorilla/inference/README.md`
  quantization section; out of scope for this reproduction fix.

---

## Files

- `gorilla/TRAINING.md` (new — reproduction guide)
- `PR-92-description.md` (new — this file)

## Closes

Closes #92
