# Known by Their Actions

> Can we fingerprint LLM agents by how they behave — not what they say?

## Overview

This project investigates whether LLM agents leave identifiable behavioral signatures in their action traces. By collecting and analyzing low-level browser interactions (keystrokes, clicks, scrolls, navigation), we train classifiers to identify:

- **Model size** (e.g., small vs. large)
- **Model family** (e.g., GPT vs. Claude vs. Gemini)
- **Granular model endpoint** (potentially with timestamp to account for model drift)

## Hypotheses

1. **Identity from actions** — Behavioral traces are sufficient to classify model size, family, reasoning level, and specific endpoint.
2. **Cross-modal identity** — Identity signals generalize across task types and domains.
3. **Modality differences** — Text(DOM)-only and multimodal agents exhibit measurably different behavioral patterns.
4. **Performance correlation** — Classifier confidence correlates with task performance metrics.
5. **Early classification** — Agent identity can be inferred from short trace prefixes; we explore the tradeoff between trace length and classification accuracy.
6. **Domain-invariant fingerprints** - do models have global browsing styles?(Train on Wikipedia → test on StackOverflow)
7. **Task-invariant fingerprints** - Is the fingerprint tied to reasoning objective? (Train on QA → test on verification)

## Approach

- Collect browser action traces from agents performing standardized tasks in a sandboxed Playwright environment
- Extract behavioral features (timing, spatial patterns, action sequences) from raw traces
- Train classifiers on aggregated feature vectors and/or sequential representations of traces
- Evaluate identity attribution accuracy across model families and sizes


## Data Plans
- Currently recasting QA questions that rely on wikipedia to track browsing behavior. We may be visiting other sites in the process of trying to answer the question.
- Can we check if detection models trained on this data on one site like wikipedia translate to other sites?
- Potential sites: StackOverflow (verification)
- Train on Wikipedia → test on StackOverflow


## DATASETS
- [2WikiMultihopQA](https://huggingface.co/datasets/framolfese/2WikiMultihopQA)
- [FRAMES](https://huggingface.co/datasets/google/frames-benchmark)
- [HotPotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)

## Supported Models
- GPT 5.4 (vision, text)
- Qwen3-VL (vision, text)
- GPT-OSS-20B (text)
- A model selection strategy might be based off of using (ScreenSpot Pro benchmark)[https://gui-agent.github.io/grounding-leaderboard/]

## Repo Structure

```
notebooks/          # Data prep and analysis
legacy/             # Early feature extraction experiments
artifacts/          # Collected trace artifacts
browser_agent_example.py #example training script
browser_worker.py
sandbox_browser.py  # Apptainer-based browser sandbox
trace_injector.py   # Trace collection and injection for Playwright
setup_browser_image.sh # Setup apptainer browser sanfbox
```
