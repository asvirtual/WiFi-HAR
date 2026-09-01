# WiFi-CSI Human Activity Recognition

Deep learning for **Human Activity Recognition (HAR)** and **Person Identification (PI)** from Wi-Fi Channel State Information (CSI).

## Overview

This project investigates device-free human sensing using commodity **80 MHz Wi-Fi Channel State Information** represented as micro-Doppler spectrograms.

The main challenge is **domain shift**: CSI representations are strongly affected by room geometry, hardware characteristics, and non-line-of-sight (NLOS) conditions. The project therefore uses task-specific deep learning architectures designed to separate human motion from environment-dependent signal variations.

The system addresses two related tasks:

- **Human Activity Recognition (HAR):** recognize activities such as walking, running, sitting, jumping, and an empty room.
- **Person Identification (PI):** identify individuals from subject-specific micro-Doppler signatures.

## Approach

### Input representation

Original CSI measurements are transformed into the time-frequency domain using **Short-Time Fourier Transform (STFT)**, producing multi-channel Doppler spectrograms.

Each input has dimensions:

```text
340 × 100 × 4
```

corresponding to time bins, Doppler-frequency bins, and four synchronized receiving antennas.

### Human Activity Recognition

The HAR pipeline uses:

- Multi-scale **Inception** convolutional backbone
- Per-window **Instance Normalization**
- **MixStyle** regularization for domain generalization
- First-order temporal differences
- **Bidirectional LSTM (BiLSTM)**
- Temporal self-attention
- Temporal variance pooling
- Late fusion across the four antenna views

### Person Identification

The PI pipeline uses a different architecture:

- Four-channel early-fusion **Inception** backbone
- No Instance Normalization, preserving localized subject-specific energy profiles
- Convolutional spatial feature extraction
- No recurrent temporal modeling

### Contrastive learning

Both tasks use **Supervised Contrastive Learning (SupCon)**. The four synchronized receiving antennas are treated as natural multi-view positive pairs, encouraging representations to retain task-relevant signatures while reducing sensitivity to environmental multipath effects.

The training objective combines:

```text
Cross-Entropy Loss + λ × Supervised Contrastive Loss
```

## Data processing

The pipeline includes:

- CSI-to-Doppler-spectrogram transformation
- Task-specific normalization / artifact removal
- Sliding-window segmentation
- Time and frequency masking with SpecAugment
- Environment- and subject-aware train/test splits
- Cross-domain evaluation on unseen environments and subjects

## Results

### Human Activity Recognition

| Evaluation | Accuracy | F1 |
|---|---:|---:|
| By day | 83.52% | 83.23% |
| By environment | 80.90% | 80.78% |
| By person | **86.79%** | **86.61%** |

### Person Identification

| Evaluation | Accuracy | F1 |
|---|---:|---:|
| In-domain | 95.10% | 91.95% |
| Cross-domain | 83.30% | 87.80% |

## Key Findings

- HAR benefits from recurrent temporal modeling with BiLSTM and temporal attention.
- Person Identification benefits from convolutional feature extraction that preserves instantaneous micro-Doppler information.
- Instance Normalization improves robustness to environment- and hardware-dependent gain variations for HAR.
- Multi-antenna supervised contrastive learning helps disentangle human signatures from environmental interference.
- Under cross-domain evaluation, **Macro F1** is a more reliable checkpointing criterion than cross-entropy loss for Person Identification.

## Technologies

Python · PyTorch · NumPy · Pandas · Matplotlib · CNNs · RNNs · BiLSTM · Attention · Supervised Contrastive Learning · Wi-Fi CSI

## Report

The project report is included in the repository.

## Authors

**Giordano Alberti**, Matteo Lazzarini, Filippo Pizzo  
Department of Information Engineering, University of Padova
