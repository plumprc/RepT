<div align="center">

# 🧭 Where Did It Go Wrong?

**Official implementation of *Attributing Undesirable LLM Behaviors via Representation Gradient Tracing***

[![arXiv](https://img.shields.io/badge/arXiv-2510.02334-b31b1b.svg)](https://arxiv.org/pdf/2510.02334)

*Zhe Li · Wei Zhao · Yige Li · Jun Sun*

</div>

---

## 📌 Overview

When an LLM produces an undesirable behavior — a harmful answer, a factual error, a backdoored
response — **which training samples caused it?**

This repository contains the code to **fine-tune LLMs**, **detect undesirable behaviors** on a test
set, and **trace each of them back to the responsible training samples** using
**RepT** (Representation Gradient Tracing) and a range of baselines.

---

## 📁 Repository Structure

```
.
├── finetune.py        # 1️⃣  LoRA fine-tune a base LLM
├── full_finetune.py   #     Full-parameter fine-tuning (alternative to LoRA)
├── generate.py        # 2️⃣  Generate responses on a test set
├── tracing.py         # 3️⃣  Trace undesirable behaviors back to training data
├── repdiff.py         # 🔎  Find the layer where representations change the most
├── utils.py           # 🛠️  Models, tokenization, representations & gradients
│
├── datasets/          # 📚  Training data + test prompts
├── lora_adapter/      # 💾  Fine-tuned LoRA adapters   (output)
├── checkpoints/       # 💾  Fully fine-tuned models    (output)
└── test_results/      # 📄  Generated responses        (output)
```

---

## ⚙️ Setup

```bash
git clone https://github.com/plumprc/RepT.git
cd RepT

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


---

## 🚀 Quick Start

The pipeline is **three steps**: fine-tune → detect → trace.

### 1️⃣ Fine-tune with LoRA

```bash
python finetune.py --dataset harmful-tuning --model llama2-7b
# → lora_adapter/llama2-7b/harmful-tuning_3
```

For full-parameter fine-tuning instead:

```bash
python full_finetune.py --dataset harmful-tuning --model llama2-7b
# → checkpoints/llama2-7b/harmful-tuning_3
```

> After full fine-tuning, point the model-loading logic in `generate.py` / `tracing.py` at your
> checkpoint.

### 2️⃣ Generate responses and collect undesirable behaviors

```bash
python generate.py --model llama2-7b --lora harmful-tuning_3 --dataset harmful-tuning_test
# → test_results/llama2-7b_harmful-tuning_3.csv
```

Inspect the generated responses, keep the ones that exhibit the undesirable behavior, and place
them under `datasets/validation/` — e.g. `datasets/validation/harmful-tuning_3_llama2-7b.csv`.

### 3️⃣ Trace them back to training samples

```bash
python tracing.py --model llama2-7b --lora harmful-tuning_3 --method RepT --topk "10 50"
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--method` | tracing method (see table below) |
| `--topk` | space-separated cutoffs, e.g. `"10 50 100"` |
| `--layer` | target layer for representation-based methods (`-1` = last) |
| `--cache` | cache extracted features to `cache/` so re-runs are fast |
| `--load_in_4bit` | 4-bit quantization (saves GPU memory) |

### 🔎 Picking a layer

`repdiff.py` reports the layer where consecutive representations differ the most, which is a good
default for `--layer`:

```bash
python repdiff.py --model llama2-7b --lora harmful-tuning_3 --p "your probe prompt"
```

---

## 🧮 Tracing Methods

`--method` supports the following (`tracing.py`):

| Method | Metric | Notes |
|---|---|---|
| **`RepT`** | cosine | representation gradient tracing (ours) |
| `TracIn` | dot | gradient dot product |
| `TracInLN` | dot | layer-normalized TracIn |
| `RapidIn` | dot | random-projected gradient |
| `LESS` | cosine | low-rank projected gradients |
| `DataInf` | — | closed-form influence |
| `LiSSA` | — | stochastic Hessian-vector products |
| `Random` | — | random ranking (sanity baseline) |
| `BM25` | — | lexical retrieval baseline |

---

## 🤖 Supported Models

Pass any of these to `--model`:

| `--model` | Checkpoint |
|---|---|
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| `llama2-7b` | Llama-2-7b-chat-hf |
| `llama2-13b` | Llama-2-13b-chat-hf |
| `llama2-70b` | Llama-2-70b-chat-hf |
| `llama3` | Meta-Llama-3.1-8B-Instruct |
| `mistral` | mistralai/Mistral-7B-Instruct-v0.3 |
| `qwen2` | Qwen/Qwen2.5-7B-Instruct |

> Model paths are resolved in `utils.get_model_name()` — edit that function to point at your own
> local checkpoints.

---

## 📝 Citation

```bibtex
@article{li2025did,
  title   = {Where Did It Go Wrong? Attributing Undesirable LLM Behaviors via Representation Gradient Tracing},
  author  = {Li, Zhe and Zhao, Wei and Li, Yige and Sun, Jun},
  journal = {arXiv preprint arXiv:2510.02334},
  year    = {2025}
}
```

---

## 🙏 Acknowledgements

We appreciate the following projects, which contributed valuable code and datasets:

- [DataInf](https://github.com/ykwon0407/DataInf)
- [LESS](https://github.com/princeton-nlp/LESS)
- [🤗 transformers](https://github.com/huggingface/transformers), [🤗 peft](https://github.com/huggingface/peft)

---

## 📮 Contact

Questions or suggestions? Open an issue, or reach out to **zheli@smu.edu.sg**.
