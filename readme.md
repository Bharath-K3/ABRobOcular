# ABRobOcular: Adversarial Benchmarking and Robustness for Ocular Biometrics

[![Paper](https://img.shields.io/badge/Paper-📄Neurocomputing-red)](https://www.sciencedirect.com/science/article/pii/S0925231225010240)
[![Dataset](https://img.shields.io/badge/Dataset-🤗%20Hugging%20Face-yellow)](https://huggingface.co/datasets/BharathK333/ABRobOcular_Attacks)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive library for evaluating and defending against adversarial attacks in ocular biometric systems.

## Abstract
Ocular biometrics, leveraging the unique traits of the eye region, stands as the pinnacle of cutting-edge technology. They have been widely adopted for identity verification and healthcare applications, and seamlessly integrated into products by tech giants like Meta and Apple. However, despite their rapid adoption and transformative potential, the vulnerability of ocular biometrics to sophisticated adversarial attacks remains alarmingly unaddressed. 

Adversarial attacks involve using digitally altered data to deceive deep learning models into producing incorrect outputs. Understanding the vulnerability of ocular biometrics against adversarial attacks and the potential of proposed defenses is vital to the robust deployment of secure and reliable technology on a global scale.

The aim of this paper is to investigate black-box and white-box-based adversarial attacks against ocular-based user recognition algorithms.
To this end, we benchmark the impact of $2$ ocular-specific, black-box adversarial attacks, and $5$ popular white-box adversarial attacks alongside their counterpart $9$ defenses across multiple datasets. In addition, we propose $3$ unified defenses against joint black-box and white-box attacks.
To the best of our knowledge, this is the first comprehensive study of its kind. To facilitate further research, we are releasing the ocular image datasets with adversarial attacks, as well as the evaluation code, under the ABRobOcular library, providing a valuable resource for future work in the field.

![alt text](assets/taxonomy.jpg)

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
```bibtex
@article{krishnamurthy2025abrobocular,
  title={ABRobOcular: Adversarial benchmarking and robustness analysis of datasets and tools for ocular-based user recognition},
  author={Krishnamurthy, Bharath and Rattani, Ajita},
  journal={Neurocomputing},
  pages={130352},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgements
This work is supported in part by the National Science Foundation (NSF) , United States award no. 2345561.
![alt text](assets/NSF_Logo.png)