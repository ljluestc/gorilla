# Gorilla Fine-tuning Guide

This document explains how to reproduce the Gorilla fine-tune on top of LLaMA using
the [FastChat](https://github.com/lm-sys/FastChat) training pipeline and the APIBench
training split (`data/apibench/huggingface_train.json`).

It was written in response to [issue #92](https://github.com/ShishirPatil/gorilla/issues/92),
which reported poor inference quality after retraining with LLaMA v2-7B.

---

## 1. Base model caveat — LLaMA v1 vs. v2

The published `gorilla-7b-hf-v0` weights were fine-tuned on top of **LLaMA v1-7B**
(Meta's original 2023 release).  LLaMA v2 uses a different default chat template
(Llama-2-chat's `[INST]`/`[/INST]` markers) and a different tokenizer vocabulary
size (32 000 → 32 000 but with added special tokens).

If you fine-tune on LLaMA v2-7B instead, you **must**:

- Use `--model_name_or_path meta-llama/Llama-2-7b-hf` (the *base*, not the `-chat`
  variant).
- Keep the FastChat `vicuna_v1.1` conversation template (the same one used for
  LLaMA v1), **not** the `llama-2` template.  The `###Instruction:`/`###Output:`
  separator style in the training data matches vicuna_v1.1, not Llama-2-chat.
- Expect a small accuracy regression versus the v0 checkpoint; the tokenizer
  mismatch means you are not building on the same embedding table.

For the best reproduction, start from `huggyllama/llama-7b` (an Apache-licensed
re-upload of the original LLaMA v1 weights).

---

## 2. Data preparation

### 2.1 Raw training file

`data/apibench/huggingface_train.json` contains one JSON object per line.  Each
object has a `"code"` field that already encodes both the instruction and the
expected output:

```
###Instruction: <natural-language question>
###Output: <<<domain>>>: ...
           <<<api_call>>>: ...
           <<<api_provider>>>: ...
           <<<explanation>>>: ...
           <<<code>>>: ...
```

### 2.2 Converting to FastChat's ShareGPT format

FastChat's `train.py` expects data in ShareGPT format — a list of conversations,
where each conversation is a list of `{"from": "human"|"gpt", "value": "..."}` turns.

Run the conversion below (save it as `scripts/convert_apibench_to_sharegpt.py`):

```python
import json, sys

def convert(src, dst):
    with open(src) as f:
        lines = [json.loads(l) for l in f if l.strip()]

    sharegpt = []
    for item in lines:
        code = item["code"]
        # Split on the first ###Output: marker
        parts = code.split("###Output:", 1)
        if len(parts) != 2:
            continue
        instruction = parts[0].replace("###Instruction:", "").strip()
        output      = parts[1].strip()
        sharegpt.append({
            "id": f"gorilla_{len(sharegpt)}",
            "conversations": [
                {"from": "human", "value": instruction},
                {"from": "gpt",   "value": output},
            ],
        })

    with open(dst, "w") as f:
        json.dump(sharegpt, f)
    print(f"Wrote {len(sharegpt)} examples to {dst}")

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
```

Usage:

```bash
python scripts/convert_apibench_to_sharegpt.py \
    data/apibench/huggingface_train.json \
    data/apibench/huggingface_train_sharegpt.json
```

---

## 3. Environment setup

```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,webui]"
pip install deepspeed
```

Tested with:

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.0.1+cu118 |
| transformers | 4.31.0 |
| deepspeed | 0.10.0 |
| FastChat | 0.2.x |

---

## 4. Training command

The following reproduces the original training hyperparameters as closely as
possible.  Run from the FastChat root.

```bash
torchrun --nproc_per_node=8 --master_port=20001 \
    fastchat/train/train.py \
    --model_name_or_path  huggyllama/llama-7b \
    --data_path           /path/to/data/apibench/huggingface_train_sharegpt.json \
    --bf16                True \
    --output_dir          /path/to/gorilla-7b-hf-ft \
    --num_train_epochs    3 \
    --per_device_train_batch_size  2 \
    --per_device_eval_batch_size   2 \
    --gradient_accumulation_steps  8 \
    --evaluation_strategy "no" \
    --save_strategy       "steps" \
    --save_steps          1200 \
    --save_total_limit    3 \
    --learning_rate       2e-5 \
    --weight_decay        0. \
    --warmup_ratio        0.03 \
    --lr_scheduler_type   "cosine" \
    --logging_steps       1 \
    --fsdp                "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
    --tf32                True \
    --model_max_length    2048 \
    --gradient_checkpointing True \
    --lazy_preprocess     True
```

> **Effective batch size** = `per_device_train_batch_size` × `gradient_accumulation_steps` × `num_gpus`  
> = 2 × 8 × 8 = **128**

These match the Vicuna v1.0 recipe on which the original Gorilla training was based.

### 4.1 Fewer GPUs

If you have fewer than 8 GPUs, increase `gradient_accumulation_steps` to keep the
effective batch size at 128.  For a single A100 (or A6000):

```bash
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 64   # 2 × 64 × 1 = 128
```

### 4.2 Using DeepSpeed instead of FSDP

Replace the `--fsdp` / `--fsdp_transformer_layer_cls_to_wrap` flags with:

```bash
--deepspeed scripts/zero3.json
```

A minimal `zero3.json`:

```json
{
  "fp16": {"enabled": false},
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "gather_16bit_weights_on_model_save": true
  },
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8,
  "steps_per_print": 10
}
```

---

## 5. Conversation template

Gorilla uses the **vicuna_v1.1** template.  When calling FastChat's `train.py` on a
LLaMA v2 base, make sure the registered conversation template resolves to
`vicuna_v1.1`.  You can verify this at the Python prompt:

```python
from fastchat.conversation import get_conv_template
print(get_conv_template("vicuna_v1.1"))
```

The system prompt should begin:
```
A chat between a curious user and an artificial intelligence assistant. ...
```

Do **not** use the `llama-2` or `llama-2-chat` template — those wrap turns in
`[INST]`/`[/INST]` brackets which will confuse the model because the training data
uses `###Instruction:`/`###Output:` markers.

---

## 6. Common pitfalls from issue #92

| Symptom | Likely cause | Fix |
|---|---|---|
| Model outputs bare text instead of `<<<api_call>>>` tags | Wrong conversation template (llama-2 instead of vicuna_v1.1) | Use `--conv-template vicuna_v1.1` at inference |
| Training loss does not decrease after epoch 1 | Learning rate too high for LLaMA v2 | Try `1e-5` |
| `###Output:` appears verbatim in responses | Data not converted to ShareGPT format; raw `code` field fed directly | Run the conversion script in §2.2 |
| Out-of-memory with 8 GPUs | `model_max_length 2048` too large for available VRAM | Reduce to `1024` or use gradient checkpointing |

---

## 7. Evaluating the fine-tuned model

After training, follow the standard Gorilla eval pipeline:

```bash
cd gorilla/eval
python get_llm_responses.py \
    --model /path/to/gorilla-7b-hf-ft \
    --output_file my_ft_hf_0shot.jsonl \
    --question_data eval-data/questions/huggingface/questions_huggingface_0_shot.jsonl \
    --api_name huggingface

cd eval-scripts
python ast_eval_hf.py \
    --api_dataset ../../../data/api/huggingface_api.jsonl \
    --apibench    ../../../data/apibench/huggingface_eval.json \
    --llm_responses ../eval-data/responses/huggingface/my_ft_hf_0shot.jsonl
```

A well-trained HuggingFace 0-shot checkpoint should score ≥ 20 % on the `ast_eval`
metric (the published `gorilla-7b-hf-v0` scores ~20.43 %).

---

## 8. References

- Gorilla paper: [arXiv:2305.15334](https://arxiv.org/abs/2305.15334)
- FastChat Vicuna training recipe: <https://github.com/lm-sys/FastChat/blob/main/docs/training.md>
- Original issue: [#92 – retrain results are poor](https://github.com/ShishirPatil/gorilla/issues/92)
