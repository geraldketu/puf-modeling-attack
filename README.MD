# Modeling and Transfer Learning Attacks Against Physical Unclonable Functions

A machine learning attack pipeline evaluating the security of PUF architectures using simulated challenge-response pairs (CRPs).

## Overview

Physical Unclonable Functions (PUFs) are hardware security primitives that exploit manufacturing variation to generate device-specific responses. This project evaluates ML-based modeling attacks against three PUF architectures:

| Architecture        | Best Model                  | Best Accuracy |
| ------------------- | --------------------------- | ------------- |
| Arbiter PUF         | Linear SVM (Xϕ)            | ~99%          |
| XOR Arbiter PUF     | MLP (Xϕ)                   | ~94%          |
| Ring Oscillator PUF | LR / SVM / MLP (RO one-hot) | >80%          |

## Repo Structure

```
puf-modeling-attack/
│
├── MLPRingOPUF (2).ipynb          # Main Ring Oscillator PUF notebook
├── README.MD                      # Project documentation
├── Ring Oscillator Experiments.ipynb
│                                   # RO-PUF experiments and analysis
│
├── arbiter_basic_models.py        # Baseline Arbiter PUF models
├── arbiter_generate_crps.py       # Arbiter PUF CRP generation
├── arbiter_mlp.py                 # MLP modeling for Arbiter PUF
├── arbiter_puf_64bit_10000crps.npz
│                                   # Saved Arbiter PUF dataset
│
├── xor_architectures.py           # XOR Arbiter PUF architectures
├── xor_basic_models.py            # Baseline XOR PUF models
├── xor_generate_crps.py           # XOR PUF CRP generation
├── xor_mlp.py                     # MLP attacks on XOR PUFs
├── xor_transfer.py                # Transfer learning experiments
├── xor_transfer2.py               # Additional transfer experiments
└── xor_vary_k.py                  # Experiments varying XOR size (k)
```

## Methods

Three feature representations were evaluated across all PUF types:

- **Xbits** — raw binary challenge bits
- **Xraw** — ±1 encoded challenge bits
- **Xϕ** — Arbiter parity/product transformation
- **RO one-hot** — one-hot encoding of selected oscillator pair indices (RO-PUF only)

Three model families were tested:

- Logistic Regression
- Support Vector Machines (Linear and RBF kernels)
- Multilayer Perceptrons (PyTorch)

## Key Findings

- **Feature representation matters more than model complexity.** A simple logistic regression with the right features outperforms a complex model with raw bits.
- **Arbiter PUFs** are highly vulnerable to linear classifiers when Xϕ is used (~99% accuracy with 10k CRPs).
- **XOR Arbiter PUFs** resist linear models but are attackable with an MLP on Xϕ (~94% accuracy with 50k CRPs).
- **Ring Oscillator PUFs** require RO one-hot features; raw bits and parity features yield near-random accuracy.
- **Transfer learning** (layer freezing) did not improve XOR Arbiter PUF attack performance in preliminary experiments.

## Requirements

```bash
pip install numpy scikit-learn torch
```

## Usage

Run experiments from within each subfolder. See individual folder READMEs (if present) or the paper for full experimental setup details.

## References

- Rührmair et al., *Modeling Attacks on Physical Unclonable Functions*, CCS 2010
- Rührmair et al., *PUFs: Myth, Fact or Busted?*, IEEE S&P 2013
- Mursi et al., *A Machine Learning Attack Resistant PUF Design*, IEEE TIFS 2018
- Ali-Pour et al., *An Efficient Approach to Model Strong PUF with MLP Using Transfer Learning*, ISQED 2022

## Ethics

This work is intended as a defensive security evaluation. All experiments were conducted on simulated PUFs. Any future work on physical hardware should only be performed with explicit authorization.
