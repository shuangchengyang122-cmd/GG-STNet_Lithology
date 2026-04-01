# GG-STNet: Gradient-Guided Spatiotemporal Network for Lithology Identification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8.0-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)

This repository provides the official TensorFlow 2.8 implementation of **GG-STNet**, a deep learning architecture designed for high-precision lithology identification in complex strata using well-log data.

## Core Innovations Included in this Code:
1. **Physical Gradient Extraction**: Explicitly calculates first-order depth derivatives from well logs to highlight bed boundaries.
2. **GG-DCNN**: Gradient-guided multi-scale dilated convolutions to capture sub-meter thin beds.
3. **DG-PhasedLSTM**: Depth-gradient phased LSTM to model long-range sedimentary context and handle transitional lithologies.
4. **Uncertainty Quantification**: Monte Carlo (MC) Dropout implementation for predictive boundary uncertainty analysis.

## Repository Structure
* `gg_stnet_model.py`: Contains the core layers and the full GG-STNet architecture.
* `train_and_evaluate.py`: The main script for training the model and executing MC Dropout inference.
* `sample_data/`: Contains a small subset of anonymized well-log data (dummy data) to verify the code execution.

## Quick Start Guide

**Step 1: Environment Setup**
Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt