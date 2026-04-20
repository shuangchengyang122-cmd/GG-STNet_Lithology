# GG-STNet: Gradient-Guided Spatiotemporal Network for Lithology Identification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8.0-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)

This repository provides the TensorFlow 2.8 implementation of **GG-STNet**, a gradient-guided spatiotemporal network developed for lithology identification from well-log data in complex coal-bearing strata.

## Main Components
1. **Log-Gradient Extraction**: Computes first-order log gradients along depth to highlight boundary-sensitive response changes.
2. **GG-DCNN**: Uses gradient-guided multi-scale dilated convolutions to enhance thin-bed boundaries and local high-frequency features.
3. **Gradient-Gated Depth-Context Modeling**: Integrates gradient cues with sequential modeling to capture vertical contextual relationships and improve discrimination of transitional lithologies.
4. **Feature Fusion and Softmax Classification**: Combines spatial and contextual representations for final lithology prediction.

## Repository Structure
- `gg_stnet_model.py`: Core layers and the full GG-STNet architecture.
- `train_gg_stnet.py`: Training and evaluation script for GG-STNet.
- `sample_data/`: A small anonymized example dataset for code verification.

## Quick Start

### 1. Environment Setup
Ensure that Python 3.8+ is installed, then install the required dependencies:

```bash
pip install -r requirements.txt
