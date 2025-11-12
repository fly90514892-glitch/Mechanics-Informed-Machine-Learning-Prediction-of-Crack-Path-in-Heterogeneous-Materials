# Mechanics-Informed Machine Learning Prediction of Crack Path in Heterogeneous Materials

This repository contains the complete implementation for the paper "Mechanics-Informed Machine Learning Prediction of Crack Path in Heterogeneous Materials" by Tengyuan Hao and Zubaer Hossain.

## Overview

Mechanics-informed ML framework for predicting crack paths in porous media. Combines FEM data with a Transformer model using physics-guided domain extraction and Variable Stiffness Boundary Condition (VSBC) for stable training.

## System Requirements

- **FEM Simulations**: MEF90
- **ML Framework**: TensorFlow v2.13+, Python 3.8+
- **Hardware**: GPU recommended for training

## Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install MEF90 (for FEM simulations)
# Follow instructions at: https://github.com/bourdin/mef90
```

## Repository Structure
```
├── FEM-generated_training_datasets/
│   ├── 631cases.zip         # Two-pore combinations (0, 1, 2 pores)
│   ├── 7176cases.zip        # Three-pore combinations (0, 1, 2, 3 pores)
│   └── generation_log.txt   # Random seeds and generation parameters
│
├── FE_code/
│   └── dp.yaml              # Phase-field FEM input file
│
├── ML_code/
│   ├── train/
│   │   └── train.py         # Training script
│   ├── predict/
│   │   ├── predict.py       # Prediction script
│   └── └── prepare.qsub     # Batch job submission script
│
├── Trained_Model/
│   ├── transformer_model_631cases/   # Model for 0, 1, 2 pores
│   └── transformer_model_7176cases/  # Model for 0, 1, 2, 3 pores
│
├── porous_medium/
│   ├── FEM/                 # .gen files for MEF90 simulations
│   └── ML/                  # Binary matrices for ML predictions
│
├── requirements.txt         # Python dependencies
└── LICENSE                  # MIT License
```

## Phase-Field FEM Parameters

The FEM simulations use the following critical parameters (from `dp.yaml`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `fracturetoughness` | 0.1 | Critical energy release rate (Gc) |
| `internalLength` | 1.25 | Regularization parameter (ℓ0) |
| `YoungsModulus` | 1.0 | Matrix elastic modulus |
| `residualstiffness` | 1e-6 | Numerical stability parameter (η) |
| `damage.type` | AT1 | Ambrosio-Tortorelli functional |
| `numstep` | 100 | Quasi-static loading steps |

### VSBC Implementation

The **Variable Stiffness Boundary Condition (VSBC)** [https://doi.org/10.1016/j.cma.2025.118414](https://doi.org/10.1016/j.cma.2025.118414) is implemented with:
- Stiffness gradient: `E_vsd(x) = (1 - x/L_x)^n * a + b`
- Parameters: `n=8, a=0.4, b=0.005`
- Domain dimensions: 200×80 (main), 200×5 (VSBC strips)

## Reproducing Paper Results
```bash
cd results_reproduction/figure_scripts/

# Figure 5: Hyperparameter optimization
python fig5_hyperparameter_tuning.py

# Figure 6: Crack paths at different porosities (0-5%)
python fig6_porosity_comparison.py --model ../../Trained_Model/transformer_model_631cases/

# Figure 14: 10% porosity extrapolation test
python fig14_extrapolation_test.py --model-631 ../../Trained_Model/transformer_model_631cases/ \
                                    --model-7176 ../../Trained_Model/transformer_model_7176cases/
```

### 4. Validate Results
```bash
# Compare ML predictions with FEM ground truth
python validate_predictions.py --tolerance 0.01

# Expected metrics (Table 1):
# - Accuracy: 0.9991
# - F1 Score: 0.9948
# - IoU: 0.9897
```

## Key Implementation Details

### Data Generation
- ROI extraction: 9×4 subdomains around crack tip
- Binary encoding: 1 = intact material, 0 = crack/pore
- Train/validation split: 80%/20% at simulation level (not ROI level)
- Random seed: 42 for all random operations

### Model Architecture
- Base: Transformer with self-attention
- Input: Flattened 36-dimensional vector
- Multi-head attention: 8 heads, key dimension 32
- Dense layer: 512 units with ReLU activation
- Dropout: 0.1 rate
- Loss function: Mean Squared Error
- Optimizer: Adam with default parameters

### Performance Benchmarks
- Training time: ~2 hours (631 cases), ~8 hours (7176 cases) on NVIDIA V100
- Prediction speed: ~1000× faster than FEM simulations
- Memory requirements: 8GB GPU memory for training

## Citation

If you use this code, please cite:
```bibtex
@article{hao2025mechanics,
  title={Mechanics-Informed Machine Learning Prediction of Crack Path in Heterogeneous Materials},
  author={Hao, Tengyuan and Hossain, Zubaer},
  journal={Journal Name},
  year={2025},
  doi={10.xxxx/xxxxx}
}
```

## Contact

For questions or issues, please contact:
- Tengyuan Hao: thao39@gatech.edu
- Zubaer Hossain: zubaer@tamu.edu

## License

MIT License - see LICENSE file for details
