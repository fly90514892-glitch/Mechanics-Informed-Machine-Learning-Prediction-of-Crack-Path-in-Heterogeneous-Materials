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
├── results_reproduction/    # Scripts to reproduce each figure
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

# Figure 5: Hyperparameter optimization
cd results_reproduction/fig5/dense
matlab -batch "run('dense.m')"
cd results_reproduction/fig5/epoches
matlab -batch "run('epoches.m')"
cd results_reproduction/fig5/key_dim
matlab -batch "run('key_dim.m')"
cd results_reproduction/fig5/num_heads
matlab -batch "run('num_heads.m')"

# Figure 6: Crack paths at different porosities (0-5%)
cd results_reproduction/fig6
matlab -batch "run('fem_crack_path.m')"
matlab -batch "run('ml_crack_path.m')"

# Figure 7: Tortuosity and the deviation from the centerline as a function of porosity percentage
cd results_reproduction/fig7
matlab -batch "run('crack_analysis.m')"

# Figure 8: IoU across different porosity levels
cd results_reproduction/fig8
matlab -batch "run('IOU_loop.m')"

# Figure 9: The ML-predicted crack path within a porous medium featuring 5% porosity at different prediction steps
cd results_reproduction/fig9
matlab -batch "run('crack_path.m')"

# Figure 10: Predicted crack paths for FE reference, baselines, and ML models
cd results_reproduction/fig10
matlab -batch "run('crack_path.m')"

# Figure 11: IoU metric for baselines and ML models
cd results_reproduction/fig11
matlab -batch "run('IOU_loop.m')"

# Figure 12: Comparison of FE and ML crack paths and corresponding IoU evolution across pore configurations and sizes
cd results_reproduction/fig12
matlab -batch "run('fem_crack_path.m')"
matlab -batch "run('ml_crack_path.m')"
matlab -batch "run('IOU_loop.m')"

# Figure 13: ML-predicted crack paths in media with 3x3 and 4x4 pores
cd results_reproduction/fig13/3x3
matlab -batch "run('ml_crack_path.m')"
cd results_reproduction/fig13/4x4
matlab -batch "run('ml_crack_path.m')"

# Figure 14: Comparison of FE and ML crack paths in 10% porous media using datasets of 631 and 7176 cases
cd results_reproduction/fig14
matlab -batch "run('ml_crack_path.m')"

# Figure 15: IoU metric for 631 and 7176 cases
cd results_reproduction/fig15
matlab -batch "run('IOU_loop.m')"

```

## Key Implementation Details

### Data Generation
- ROI extraction: 9×4 subdomains around crack tip
- Binary encoding: 0 = intact material, 1 = crack/pore
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

## Contact

For questions or issues, please contact:
- Tengyuan Hao: thao39@gatech.edu
- Zubaer Hossain: zubaer@tamu.edu

## License

MIT License - see LICENSE file for details
