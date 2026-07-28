#!/usr/bin/env python3
"""
Convert an APIBench JSONL training file (huggingface_train.json, torchhub_train.json,
tensorflow_train.json) to the ShareGPT conversation format expected by FastChat's
train.py.

Usage:
    python scripts/convert_apibench_to_sharegpt.py \
        data/apibench/huggingface_train.json \
        data/apibench/huggingface_train_sharegpt.json

Each line in the source file must be a JSON object with at least a "code" key whose
value looks like:
    ###Instruction: <question>\n###Output: <<<domain>>>: ...\n<<<api_call>>>: ...

The output is a single JSON array of ShareGPT conversation objects:
    [{"id": "gorilla_0", "conversations": [
        {"from": "human", "value": "<question>"},
        {"from": "gpt",   "value": "<<<domain>>>: ...\n<<<api_call>>>: ..."}
    ]}, ...]
"""

import argparse
import json
import sys
from pathlib import Path


# Some lines use '### Instruction:' / '### Output:' (with a space after ###)
INSTRUCTION_MARKERS = ("###Instruction:", "### Instruction:")
OUTPUT_MARKERS = ("###Output:", "### Output:")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert APIBench train JSONL → FastChat ShareGPT JSON"
    )
    parser.add_argument("src", type=Path, help="Source APIBench .json file")
    parser.add_argument("dst", type=Path, help="Output ShareGPT .json file")
    parser.add_argument(
        "--id-prefix",
        default="gorilla",
        help="Prefix for conversation IDs (default: gorilla)",
    )
    parser.add_argument(
        "--skip-malformed",
        action="store_true",
        help="Skip lines that don't contain ###Output: instead of aborting",
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    records = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append((lineno, json.loads(line)))
            except json.JSONDecodeError as exc:
                print(f"[WARN] line {lineno}: JSON parse error — {exc}", file=sys.stderr)
    return records


def convert(records, id_prefix: str, skip_malformed: bool):
    conversations = []
    skipped = 0

    for lineno, item in records:
        code = item.get("code", "")

        output_marker = next((m for m in OUTPUT_MARKERS if m in code), None)
        if output_marker is None:
            msg = f"[WARN] line {lineno}: no '###Output:' marker found — skipping"
            if skip_malformed:
                print(msg, file=sys.stderr)
                skipped += 1
                continue
            else:
                raise ValueError(msg)

        before, after = code.split(output_marker, 1)
        instruction = before
        for marker in INSTRUCTION_MARKERS:
            instruction = instruction.replace(marker, "")
        instruction = instruction.strip()
        output = after.strip()

        if not instruction or not output:
            msg = f"[WARN] line {lineno}: empty instruction or output — skipping"
            if skip_malformed:
                print(msg, file=sys.stderr)
                skipped += 1
                continue
            else:
                raise ValueError(msg)

        conversations.append(
            {
                "id": f"{id_prefix}_{len(conversations)}",
                "conversations": [
                    {"from": "human", "value": instruction},
                    {"from": "gpt", "value": output},
                ],
            }
        )

    return conversations, skipped


def main():
    args = parse_args()

    if not args.src.exists():
        print(f"[ERROR] source file not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    args.dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.src} …")
    records = load_jsonl(args.src)
    print(f"  {len(records)} records loaded")

    conversations, skipped = convert(records, args.id_prefix, args.skip_malformed)

    with args.dst.open("w", encoding="utf-8") as fh:
        json.dump(conversations, fh, ensure_ascii=False, indent=2)

    print(f"Wrote {len(conversations)} conversations to {args.dst}")
    if skipped:
        print(f"  ({skipped} records skipped due to malformed entries)")


if __name__ == "__main__":
    main()
