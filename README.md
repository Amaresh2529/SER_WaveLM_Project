# 🎙️ Microphone-Blind Speech Emotion Recognition (SER)

> **A Domain-Adversarial Architecture utilizing Microsoft WaveLM, Hierarchical Attention, and Gradient Reversal to eliminate acoustic mismatch in real-world audio environments.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![WaveLM](https://img.shields.io/badge/Model-WaveLM-lightgrey)

## 📌 Project Overview
Standard Speech Emotion Recognition (SER) models suffer from the "Domain Gap" trap: they memorize the background noise, room echo, and specific microphones of their training datasets instead of learning true human emotion. When deployed in the real world on completely unseen audio, these models experience severe performance degradation.

This project introduces a robust, cross-corpus architecture designed to be strictly **"Microphone-Blind."** By employing a Domain-Adversarial Neural Network (DANN) framework via a Gradient Reversal Layer (GRL), the model mathematically scrubs acoustic background noise, forcing the foundation model to focus exclusively on emotional acoustic markers.

## 🏗️ The Architecture (The 3 Pillars)

This system processes raw `.wav` audio files (16 kHz) through an end-to-end deep learning pipeline without relying on legacy spectrogram image conversion:

1. **Pillar 1: Feature Extraction (Microsoft WaveLM)**
   - Raw audio is fed into the pre-trained WaveLM foundation model.
   - Outputs a highly detailed 768-dimensional acoustic feature matrix, taking microscopic snapshots of the voice every 20 milliseconds.
2. **Pillar 2: Temporal Focus (Hierarchical Attention Pooling)**
   - Since human emotion is not constant (e.g., sudden shouts amidst silence), an attention mechanism acts as a mathematical "spotlight."
   - It assigns near-zero weights to silence and amplifies emotional spikes, compressing the varying-length audio into a single Context Vector.
3. **Pillar 3: Domain-Adversarial Filtering (Gradient Reversal Layer)**
   - The pipeline splits into an **Emotion Classifier** and a **Domain (Microphone) Classifier**.
   - A Gradient Reversal Layer (GRL) is placed before the Domain Classifier. If the model successfully recognizes the background noise, the GRL flips the gradient and mathematically punishes the network. 
   - **Result:** The AI scrambles and deletes domain-specific noise to avoid punishment, achieving zero-shot microphone independence.

## 📊 Key Results (Ablation Study)

To prove the efficacy of the architecture, we evaluated the model using a strict **Leave-One-Corpus-Out (LOCO) protocol**, completely isolating the test dataset (RAVDESS) during training to simulate a zero-shot real-world deployment.

| Model Architecture | Training Accuracy | Unseen Validation (RAVDESS) |
| :--- | :---: | :---: |
| **Baseline** (WaveLM + Attention, *No GRL*) | 85.58% | 42.57% |
| **Proposed** (WaveLM + Attention + *GRL*) | **~ 85.00%** | **48.49%** |

**Conclusion:** Disabling the GRL caused the baseline model to overfit to the training microphones. Activating the Domain-Adversarial GRL successfully scrubbed the acoustic bias, yielding a **~6% absolute improvement** in unweighted average recall (UAR) in entirely unseen environments.

## 📂 Repository Structure
```text
├── data/                   # Directory for raw datasets (CREMA-D, RAVDESS, TESS, SAVEE)
├── src/                    # Source code
│   ├── models/             # PyTorch model definitions (WaveLM, Attention, GRL)
│   ├── utils/              # Data loaders and audio processing scripts
│   ├── train.py            # Main training script (with GRL)
│   └── train_ablation.py   # Baseline training script (GRL disabled)
├── results/                # Saved model weights and training logs
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
