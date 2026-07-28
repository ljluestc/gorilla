# HOWTO: Fine-tune llama-2-7b in Azure AI Studio

> ⚠️ See the **Azure RAFT Distillation Recipe repo** ([Azure-Samples/raft-distillation-recipe](https://aka.ms/raft-recipe)) for instructions, notebooks and infrastructure provisioning for Meta Llama 3.1 and 3.2 as well as GPT-4o.

## Prerequisites

[Prerequisites in MS Learn article "Fine-tune a Llama 2 model in Azure AI Studio"](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/fine-tune-model-llama#prerequisites)

## Key things to get right for everything to work

- Select the West US 3 location
- Use a Pay As You go Subscription with a credit card linked
- Make sure the subscription is registered to the `Microsoft.Network` resource provider

## Detailed step by step

This builds on the ["Fine-tune a Llama 2 model in Azure AI Studio" MS Learn tutorial](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/fine-tune-model-llama#prerequisites) and adds a few details here and there.

Open https://ai.azure.com/

Create a new AI Project
![Step 01](images/azure-ai-studio-finetuning-01.png)

Enter a name and create a new resource
![Step 02](images/azure-ai-studio-finetuning-02.png)

Enter an AI Hub resource name, select the PAYG (Pay As You Go) Subscription and West US 3 location
![Step 03](images/azure-ai-studio-finetuning-03.png)

Note: It's important to use a PAYG subscription with a credit card linked to the account. Grant based subscriptions and credits will not work.

Review that the location is correctly set to West US 3 and that the subscription is correct
![Step 04](images/azure-ai-studio-finetuning-04.png)

The resources should begin being created
![Step 05](images/azure-ai-studio-finetuning-05.png)

Wait until all resources have been created
![Step 06](images/azure-ai-studio-finetuning-06.png)

Once in the AI Studio project, open the Fine-tuning tab and click on the Fine-tune model button
![Step 07](images/azure-ai-studio-finetuning-07.png)

Select the model to fine-tune, for example Llama 2 7b
![Step 08](images/azure-ai-studio-finetuning-08.png)

Subscribe if necessary to the Meta subscription and start the fine-tuning
![Step 09](images/azure-ai-studio-finetuning-09.png)

Enter the name of the fine-tuned model
![Step 10](images/azure-ai-studio-finetuning-10.png)

Select the task type, currently, only text generation is supported
![Step 11](images/azure-ai-studio-finetuning-11.png)

Select the upload data option and upload your file, it must be in JSONL format
![Step 12](images/azure-ai-studio-finetuning-12.png)

The wizard will show you an overview of the top lines
![Step 13](images/azure-ai-studio-finetuning-13.png)

Select which columns is the prompt and which one is the completion column
![Step 14](images/azure-ai-studio-finetuning-14.png)

Select the task parameters
![Step 15](images/azure-ai-studio-finetuning-15.png)

Review the settings
![Step 16](images/azure-ai-studio-finetuning-16.png)

The job should be in running state
![Step 17](images/azure-ai-studio-finetuning-17.png)

Wait until the job is completed
![Step 18](images/azure-ai-studio-finetuning-18.png)

## Finding the endpoint credentials after deployment

Once the fine-tuned model is deployed, you need three values from Azure AI Studio before `raft/eval.py` can call it:

| Value | Meaning | Where to find it |
| --- | --- | --- |
| `base_url` | The endpoint URL the OpenAI client targets | **Deployments / Endpoints** tab in the AI Studio project |
| `api_key` | Authentication token (or use keyless via `AZURE_OPENAI_AD_TOKEN` / `DefaultAzureCredential`) | Shown next to the deployment |
| `model_name` | Either the Azure OpenAI deployment name, or the model id of a serverless MaaS deployment | Shown next to the deployment; for MaaS it is the **Model name** field |

The exact shape of `base_url` depends on which deployment "flavor" you used. There are two flavors supported by `raft/eval.py` together with `raft/client_utils.py`:

### Flavor A — Azure OpenAI Service

This is the most common case for Llama-2 fine-tuning created through the AI Studio wizard. The endpoint is owned by an `Azure OpenAI` resource attached to your AI Hub.

1. Open https://ai.azure.com/ and switch into your project.
2. In the left navigation, open **Build and deploy** → **Deployments** (older UI showed this under **Model + Endpoints**).
3. Click the deployment row for your fine-tuned model (e.g. `gorilla-ft`).
4. The **Target URI** field has the form:

   ```
   https://<resource>.openai.azure.com/deployments/<deployment-name>
   ```

5. Paste it into your `.env` and **strip the path** so that only the resource domain remains. As noted in `raft/README.md`, the SDK appends the rest for you:

   ```env
   AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
   AZURE_OPENAI_API_KEY=<key>
   AZURE_OPENAI_DEPLOYMENT=<deployment-name>
   OPENAI_API_VERSION=2023-05-15
   ```

6. Pass `--model <deployment-name>` to `eval.py`. The deployment name is the value you typed in step 10 of the wizard — it is also shown verbatim under the deployment row in AI Studio.

### Flavor B — Azure AI Studio Model-as-a-Service (serverless)

If you instead deployed as a "serverless endpoint" (visible in AI Studio under **Deployments** with the **Endpoint type** set to *Serverless API*), the deployment exposes an OpenAI-compatible REST URL that already includes `/v1`. `eval.py` consumes it through the `EVAL_OPENAI_BASE_URL` env var (default `--env-prefix EVAL`).

1. In the AI Studio project, go to **Deployments** → click your deployment.
2. Open **Consume** (or **View code**). The **Target URI** has the form:

   ```
   https://<deployment>-<hash>.serverless.<region>.inference.ai.azure.com/v1
   ```

3. Copy the full URL — including the trailing `/v1` — and paste it as `EVAL_OPENAI_BASE_URL`. Then add the key and model:

   ```env
   EVAL_OPENAI_BASE_URL=https://<deployment>-<hash>.serverless.<region>.inference.ai.azure.com/v1
   EVAL_OPENAI_API_KEY=<key>
   EVAL_MODEL=<model-id>
   ```

   `EVAL_MODEL` matches the **Model name** field shown on the deployment row, e.g. `Meta-Llama-3-70B-Instruct`.

4. Run `python3 eval.py --question-file … --answer-file … --model "$EVAL_MODEL" --env-prefix EVAL`.

See [`raft/README.md` § "Configuring different endpoints for the completion and embedding models"](../../README.md#configuring-different-endpoints-for-the-completion-and-embedding-models) for the full set of `COMPLETION_*` / `EMBEDDING_*` / `EVAL_*` overrides supported by `client_utils.py`.

### Why the two flavors exist

`raft/client_utils.py::is_azure()` switches behaviour based on whether `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_KEY` / `AZURE_OPENAI_AD_TOKEN` is present in the environment (Flavor A). If none of those are set, the client falls back to the OpenAI SDK and reads the plain `EVAL_OPENAI_BASE_URL` (Flavor B). This is why `eval.py` works for both shapes without any code change — you only need to set the right env vars.
