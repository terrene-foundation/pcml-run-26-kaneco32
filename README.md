# PCML Run 26 — ML Foundations for Professionals

Terrene Open Academy course powered by the [Kailash Python SDK](https://github.com/terrene-foundation/kailash-py).

From zero Python to production ML engineering across 6 modules (192 contact hours).

## Quick Start

```bash
# 1. Clone (includes datasets via Git LFS — requires git-lfs installed)
git clone https://github.com/terrene-foundation/pcml-run26-2601.git
cd pcml-run26-2601

# 2. Create your working branch
git checkout -b your-name/work

# 3. Set up Python environment
uv sync

# 4. Create your .env file
cp .env.example .env
# Edit .env with your API keys (needed for M6)

# 5. Run your first exercise
uv run python modules/mlfp01/local/ex_1/01_*.py
```

## Requirements

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git LFS** — `brew install git-lfs` (macOS) or `apt install git-lfs` (Linux)
- **GPU recommended** for M5-M6 (MPS on Apple Silicon, CUDA on Linux/Windows)

## Course Structure

| Module | Title | Hours |
|--------|-------|-------|
| MLFP01 | Data Pipelines and Visualisation Mastery with Python | 25h |
| MLFP02 | Statistical Mastery for ML and AI Success | 25h |
| MLFP03 | Supervised ML for Building and Deploying Models | 25h |
| MLFP04 | Unsupervised ML and Advanced Techniques | 25h |
| MLFP05 | Deep Learning and Vision | 25h |
| MLFP06 | Language Models and Agentic Workflows | 25h |

**Foundation Certificate**: Modules 1-4 (128 hours)
**Advanced Certificate**: Modules 5-6 (64 hours)

## How to Work

### Directory Layout

```
data/                          # Datasets (auto-fetched via Git LFS)
modules/mlfpNN/
  readings/                    # Deck PDF, textbook PDF
  local/ex_N/                  # Exercise files — YOUR WORKSPACE
    helpers.py                 # Shared utilities (provided, do not edit)
    01_technique.py            # Fill in the TODO blanks
    02_technique.py
    ...
  colab/ex_N/                  # Google Colab notebooks (alternative)
shared/                        # Python utilities (installed via uv sync)
```

### Exercise Structure

Each exercise is a directory. Each file covers **one technique** with a complete narrative:

1. **Theory** — Why this technique exists (read this first)
2. **Build** — Implement the model (fill in `# TODO:` blanks)
3. **Train** — Train and track with kailash-ml
4. **Visualise** — See the model's actual behaviour
5. **Apply** — Solve a real-world business problem

Look for `____` placeholders — that's where your code goes. The `# TODO:` comment above each blank tells you what to write.

### Running Exercises

```bash
# From the repo root:
uv run python modules/mlfp05/local/ex_1/01_standard_ae.py

# Or activate the venv first:
source .venv/bin/activate
python modules/mlfp05/local/ex_1/01_standard_ae.py
```

### Committing Your Work

```bash
git add modules/mlfp05/local/ex_1/01_standard_ae.py
git commit -m "Complete ex_1.01: Standard Autoencoder"
git push
```

## Reading Materials

Each module includes PDF readings in `modules/mlfpNN/readings/`:
- **deck.pdf** — Full lecture deck
- **textbook.pdf** — Textbook chapter with derivations
- **notes.pdf** — Speaker notes with timing cues

## Kailash Platform

All frameworks are pre-installed via `uv sync`:

| Framework | Purpose |
|-----------|---------|
| kailash | Workflow orchestration, 140+ nodes |
| kailash-ml | 13 ML engines (polars-native) |
| kailash-dataflow | Zero-config database operations |
| kailash-kaizen | AI agent framework |
| kailash-pact | Governance (D/T/R) |
| kailash-align | LLM fine-tuning |

## Troubleshooting

**`ModuleNotFoundError: No module named 'shared'`** — Run `uv sync` from the repo root first.

**`git clone` hangs or is slow** — Install Git LFS: `git lfs install`, then re-clone.

**GPU not detected** — Check `python -c "import torch; print(torch.backends.mps.is_available())"` (Mac) or `torch.cuda.is_available()` (Linux).

## License

Apache 2.0 — [Terrene Foundation](https://terrene.foundation)
