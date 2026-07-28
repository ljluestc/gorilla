# Docs: explain how to find `base_url` in Azure AI Studio for evaluating a fine-tuned Gorilla RAFT model

Closes #322.

## Summary

Issue #322 asks how to find the `base_url` of a fine-tuned model in Azure AI Studio so that `raft/eval.py` can be run against it. The repo currently has the answer implicit (`raft/README.md` § "7. Evaluate RAFT model" tells the user to "fill in `base_url`, `api_key` and `model_name` in `eval.py`, these can be found in the AI Studio") but never explains *where* to look. This PR adds that missing guidance, covering both the Azure OpenAI Service deployment and the AI Studio serverless (Model-as-a-Service) deployment.

## Why

The reproduction path in `raft/azure-ai-studio-ft/howto.md` ends after the fine-tuning job completes (step 18). The reader is then dropped into `raft/README.md` § "7" with no visual aid or end-to-end flow for retrieving the endpoint credentials — exactly what issue #322 reports.

`raft/client_utils.py::is_azure()` switches behaviour based on which env vars are present (Flavor A — `AZURE_OPENAI_*` — vs Flavor B — plain `EVAL_OPENAI_BASE_URL`). Both shapes need to be documented so users on either deployment model can find the right `base_url`.

## Changes

### `raft/azure-ai-studio-ft/howto.md`

New section **"Finding the endpoint credentials after deployment"** appended after step 18. Contents:

- Table summarising the three values (`base_url`, `api_key`, `model_name`) and where each lives in AI Studio.
- **Flavor A — Azure OpenAI Service**: how to copy the Target URI, strip the path, which env vars to set, what to pass to `--model`.
- **Flavor B — serverless MaaS**: how to copy the full `/v1` URL, which env vars to set, the right flag set (`--env-prefix EVAL`).
- Short note pointing back to `raft/README.md` § "Configuring different endpoints…" for the full env-var matrix.
- Explanation of why two flavors exist (the `is_azure()` switch in `client_utils.py`).

### `raft/README.md`

Section "7. Evaluate RAFT model" is updated to:

- Link explicitly to the new section in `howto.md`.
- Spell out which env vars / flags to use for the Azure OpenAI vs MaaS deployment, with copy-pasteable `.env` snippets.

### `PR-322-description.md` (new)

This file. Kept in the working tree so a maintainer can copy it into the GitHub PR description; safe to delete once the PR is open.

## How I verified

I read the relevant code paths and confirmed the documented behaviour matches what the client expects:

- `raft/eval.py` calls `build_openai_client(env_prefix=args.env_prefix)` (default `EVAL`) — confirmed at line 55.
- `raft/client_utils.py` reads `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_API_VERSION` and constructs an `AzureOpenAI` client. The path-strip notice ("make sure you strip the path from the endpoint and keep just the domain") is preserved from the existing README.
- `raft/README.md:110` already documents the `COMPLETION_OPENAI_BASE_URL=https://Meta-Llama-3-70B-Instruct-<replace_me>-serverless.eastus2.inference.ai.azure.com/v1` pattern for MaaS endpoints; the new section reuses the same URL shape for consistency.
- `raft/README.md:256` previously pointed users to a steps blob in AI Studio without naming which tab to open; the new section names the tab ("Deployments / Endpoints").

No code changes — pure documentation. No CI or build changes.

## Risks

Low. Doc-only change. The new section explicitly states that path-strip semantics (`AZURE_OPENAI_ENDPOINT` must be `<resource>.openai.azure.com/`, not the deployment-rest URL) match the existing README note, so there is no behaviour drift for existing users.

## Out of scope

- Adding screenshots. The existing `howto.md` uses steps 01–18 with inline screenshots; the new text section deliberately does not add screenshots (they would belong in the same `images/` directory but require access to the AI Studio UI, which is not available to automation). The text is written so that it works without screenshots.
- Updating the upstream [`Azure-Samples/raft-distillation-recipe`](https://aka.ms/raft-recipe) repo. The top of `howto.md` already points at that repo for Meta Llama 3.1/3.2 and GPT-4o, but it lives outside this repository.

## Files

- `raft/azure-ai-studio-ft/howto.md` (modified — new section appended)
- `raft/README.md` (modified — § "7" expanded)
- `PR-322-description.md` (new — this file)

## Closes

Closes #322
