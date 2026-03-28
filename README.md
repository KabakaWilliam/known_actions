# Known by Their Actions

> Can we fingerprint LLM agents by how they behave — not what they say?

## Overview

This project investigates whether LLM agents leave identifiable behavioral signatures in their action traces. By collecting and analyzing low-level browser interactions (keystrokes, clicks, scrolls, navigation), we train classifiers to identify:

- **Model size** (e.g., small vs. large)
- **Model family** (e.g., GPT vs. Claude vs. Gemini)
- **Granular model endpoint** (potentially with timestamp to account for model drift)

## Hypotheses

1. **Identity from actions** — Behavioral traces are sufficient to classify model size, family, and specific endpoint.
2. **Cross-modal identity** — Identity signals generalize across task types and domains.
3. **Modality differences** — Text-only and multimodal agents exhibit measurably different behavioral patterns.
4. **Performance correlation** — Classifier confidence correlates with task performance metrics.
5. **Early classification** — Agent identity can be inferred from short trace prefixes; we explore the tradeoff between trace length and classification accuracy.

## Approach

- Collect browser action traces from agents performing standardized tasks in a sandboxed Playwright environment
- Extract behavioral features (timing, spatial patterns, action sequences) from raw traces
- Train classifiers on aggregated feature vectors and/or sequential representations of traces
- Evaluate identity attribution accuracy across model families and sizes

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
