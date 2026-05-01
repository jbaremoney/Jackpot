# JACKPOT!

### Reliable Extraction of Strong Lottery Ticket Subnetworks from Untrained Neural Networks

Official implementation of the paper  
**[JACKPOT!: Reliable Extraction of Strong Lottery Ticket Subnetworks from Untrained Neural Networks](LINK)**

![Accuracy Plot](imgs/multibar_real.png)

## Repository Structure

```text
src/wtl/
├── models/       # model definitions and masking wrappers
├── pruning/      # popup, SNIP, GraSP, IMP
├── training/     # training and evaluation loops
└── utils/        # reproducibility and metric helpers

scripts/          # runnable experiment entry points
notebooks/        # paper figure reproduction notebooks
tests/            # lightweight correctness tests
```

## 1. Requirements

This repository was developed with Python 3.10.

### 1.1. Clone the repository

```bash
git clone https://github.com/jbaremoney/Jackpot
cd Jackpot
```

### 1.2. Create and activate a virtual environment

Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.3. Install dependencies

```bash
pip install -r requirements.txt
```

### 1.4. Dataset setup

The experiments in this repository primarily use MedMNIST and CIFAR10 datasets.

If you are using the provided training / experiment scripts with `torchvision`, the datasets will be downloaded automatically the first time they are needed. By default, you may wish to store datasets under:

```bash
./data
```

If your local setup uses a different data path, update the corresponding script or configuration file accordingly.

## 2. Experiment

### 2.1. Train

```bash
python scripts/run_dbl_popup.py
```

### 2.2. Evaluate
Scripts generally contain code to evaluate a trained model already. The function used to evaluate a model on test data is at the following location: 
```bash
./src/Jackpot/training/eval.py
```


### 2.3. Reproduce Paper Figures

Code used to generate figures from the paper is provided in:

```bash
experiments/paper_figures/
```

## 3. Pretrained Models
A few pretrained models can be downloaded directly at this LINK. 
The models for download are as follows:


### Notes

* Results, checkpoints, and generated outputs may be saved under `results/`.
* For reproducibility, random seeds can be set using the utilities in `src/wtl/utils/seed.py`.
* This repository was tested primarily on macOS/Linux. Minor path adjustments may be needed on Windows.


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 

Run scripts from the repo root with `PYTHONPATH` including `src`:

```bash
PYTHONPATH=src python scripts/run_dbl_popup.py
```

# It's already in there
[![Surf's Up Scene](imgs/surfs_up_vid_thumbnail.png)](https://youtu.be/VD8UttNfU60?si=N8Iir_yeo5GhBDXs&t=30)
