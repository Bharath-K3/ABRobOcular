# ABRobOcular: Adversarial Benchmarking and Robustness for Ocular Biometrics

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive library for evaluating and defending against adversarial attacks in ocular biometric systems.

## Overview
ABRobOcular provides implementations of both black-box and white-box adversarial attacks, along with various defense mechanisms, specifically designed for ocular-based user recognition systems.

## Key Features
- Implementation of 5 gradient-based white-box attacks
- Novel black-box adversarial attacks:
  - Patch Occlusion Attack
  - Monocle Blending Attack
- Evaluation framework for 5 benchmark datasets
- 9 state-of-the-art white-box defenses
- Unified defense mechanisms for joint protection
- Multi-detector model for real-time attack detection
- TensorFlow and PyTorch support

## Directory Structure
```plaintext
ABRobOcular/
├── AB_Rob_Ocular/
│   ├── AB_Rob_Ocular_Tensor.py
│   ├── AB_Rob_Ocular_Torch.py
│   ├── extras.py
│   └── __init__.py
├── Adversarial_Images/
├── Black-Box Defense/
│   ├── Augmentation_Strip_ArcFace.ipynb
│   ├── Patch_Visualization.ipynb
│   ├── Strip_Visualization.ipynb
├── Detection_and_Attribution/
│   ├── Complete_DA_Network.ipynb
│   ├── dataset_loader.py
│   ├── train.py
│   ├── eval.py
│   └── predict.py
├── ocular_recognition/
│   ├── train/
│   ├── test/
│   └── validation/
├── trained_models/
├── Base_Model.ipynb
├── config.yml
├── dataset_loader.py
├── eval.py
├── fgsm_sample_run.py
├── predict.py
├── requirements.txt
└── train.py
```

## Installation
To install ABRobOcular, follow these steps:
```bash
git clone https://github.com/yourusername/ABRobOcular.git
cd ABRobOcular
pip install -r requirements.txt
```

## Citation
Please cite us if you find our work useful
'''bibtex
@article{abrobocular2025,
  title={ABRobOcular: Adversarial Benchmarking and Robustness for Ocular Biometrics},
}# ABRobOcular
```
