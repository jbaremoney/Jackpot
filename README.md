# Jackpot

Research code for **finding strong lottery-ticket subnetworks inside randomly initialized (untrained) networks**, and comparing that to classic pruning-at-initialization baselines.

## What’s here

| Area | Role |
|------|------|
| `src/Jackpot/pruning/popup.py` | **Edge-popup** style trainable scores; binary masks via a straight-through estimator. |
| `src/Jackpot/pruning/snip.py`, `grasp.py` | **SNIP** / **GraSP**: one-shot masks from gradients / Hessian–gradient products on a mini-batch. |
| `src/Jackpot/pruning/imp.py` | **Iterative magnitude pruning** on a masked copy of the network (trained baseline). |
| `src/Jackpot/models/` | CIFAR-friendly **VGG-16** and **MLP**, plus `MaskLayer` / `MaskedNetwork` for explicit binary masks. |
| `src/Jackpot/training/` | Data loaders, training loop, and eval helpers. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run scripts from the repo root with `PYTHONPATH` including `src`:

```bash
PYTHONPATH=src python scripts/run_dbl_popup.py
```
