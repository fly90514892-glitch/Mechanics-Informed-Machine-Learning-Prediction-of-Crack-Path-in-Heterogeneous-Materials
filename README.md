# Mechanics-Informed Machine Learning Prediction of Crack Path in Heterogeneous Materials

Mechanics-informed ML framework for predicting crack paths in porous media. Combines FEM data with a Transformer model using physics-guided domain extraction and Variable Stiffness Boundary Condition (VSBC) for stable training. Provides fast, accurate, and generalizable alternatives to costly FEM simulations.

## Repository Structure

### FEM-generated_training_datasets/
Contains the FEM simulation datasets used for training the Transformer models.  
- `631cases.zip` → 631 cases with **two-pore** combinations.  
- `7176cases.zip` → 7176 cases with **three-pore** combinations.  
- Example format:
  - Input matrix: `8_map0.txt` (initial microstructure and damage field)
  - Output matrix: `8_map1.txt` (final crack path after propagation)
  - `8` is the case number

**Dataset Generation:** All datasets were generated deterministically using MEF90 with fixed parameters specified in `FE_code/dp.yaml`. The random seeds for pore placement are documented in `porous_medium/generation_log.txt`.

### FE_code/
Contains scripts and configuration files for running FEM simulations with MEF90.  

**Key Files:**
- `dp.yaml`: Complete MEF90 input file specifying all phase-field parameters
  - Phase-field model: AT1 damage model
  - Fracture toughness: `cs0001: 0.1` (matrix), `cs0002-cs0051: 100.0` (pores)
  - Young's modulus: Spatially varying (see `cs0002-cs0051` sections)
  - Internal length scale: `l0 = 2.0`
  - Residual stiffness: `1e-6`
  - Poisson's ratio: `0.3` (matrix), `0.001` (pores)
  - Time stepping: 100 quasi-static steps from t=0 to t=1
  
- **VSBC Implementation:**
  - Boundary sections `vs0001` and `vs0002` define the variable stiffness boundary condition
  - `boundaryDisplacement: 0,±40.0000,0` applies symmetric displacement loading
  - `boundaryDisplacement_scaling: linear` specifies the scaling profile
  - Stiffness gradient is achieved through the spatially varying Young's modulus profile (`cs0002-cs0051`) following Eq. (3) in the paper
  - The exponentially decaying modulus values (from 0.3453 to 0.005) create the self-adjusting load concentration ahead of the crack tip

**Mesh Details:**
- Element type: 4-node quadrilateral elements (Q1)
- Mesh refinement: Uniform element size of 0.5 units
- Domain dimensions: 160 × 80 units (as specified in Section 3.1 of the paper)

### ML_code/
Contains the machine learning training and prediction scripts.  

**Directory Structure:**
- `train/`: Transformer training pipeline
  - `train.py`: Main training script with hyperparameters documented
  - `data_loader.py`: Loads and preprocesses FEM datasets
  - `model_architecture.py`: Defines the Transformer architecture
  - `config.yaml`: All hyperparameters (learning rate, batch size, etc.)
  
- `predict/`: Inference pipeline
  - `prepare.qsub`: Job submission script for prediction
  - `predict.py`: Loads trained model and generates crack path predictions
  - `postprocess.py`: Converts predictions to visualization format

**Model Hyperparameters (documented in `ML_code/train/config.yaml`):**
- Learning rate: 1e-4
- Batch size: 32
- Number of epochs: 100
- Transformer layers: 6
- Attention heads: 8
- Embedding dimension: 256
- Dropout rate: 0.1
- Optimizer: Adam
- Loss function: Binary cross-entropy

**Random Seeds for Reproducibility:**
All random operations use fixed seeds:
```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
```

### Trained_Model/
Contains trained Transformer models with full training metadata.

- `transformer_model_631cases/`: Model trained on 631 two-pore cases
  - Training history: `training_history.json`
  - Model weights: `model_weights.h5`
  - Configuration: `model_config.json`
  
- `transformer_model_7176cases/`: Model trained on 7176 three-pore cases
  - **Note:** Model size exceeds GitHub's upload limit
  - Download from Dropbox: [https://www.dropbox.com/scl/fi/5nqqvj2hvbeo9vjjch6ah/transformer_model_7176cases.zip?rlkey=b7lsaqd0kgw8vzmy1le36iuwl&dl=0]
  - Training took 48 hours on NVIDIA A100 GPU
  - Final validation accuracy: 94.2%

### porous_medium/
Contains FEM and ML domains of porous materials used in the simulations.

**Structure:**
- `FEM/*.gen`: Genesis mesh files for MEF90 simulations
- `ML/*.txt`: Preprocessed domains for ML prediction
- `generation_log.txt`: Documents pore placement parameters and random seeds
- `README_porous_medium.md`: Detailed description of domain generation procedure

### Documentation Files
- `LICENSE`: Open-source license (MIT)
- `INSTALL.md`: Step-by-step installation instructions for all dependencies
- `TUTORIAL.md`: Complete workflow from data generation to prediction
- `README.md`: This file

---

## Complete Reproducibility Workflow

### Prerequisites
```bash
# Install MEF90 (Phase-field FEM solver)
git clone https://github.com/bourdin/mef90.git
cd mef90 && make install

# Install Python dependencies
pip install tensorflow==2.12.0 numpy==1.23.5 scipy matplotlib

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Step 1: Generate FEM Training Data (Optional - data provided)

To regenerate the training datasets from scratch:
```bash
cd FE_code

# Generate a single case with pore configuration #8
mef90 -i dp.yaml -mesh ../porous_medium/FEM/case_8.gen -output results/case_8.vtu

# Batch generation (631 cases)
python generate_631_cases.py  # Uses seed=42 for pore placement

# Extract damage field for ML training
python extract_damage_fields.py --input results/ --output ../FEM-generated_training_datasets/631cases/
```

**Expected Output:**
- Each case produces two files: `{case_id}_map0.txt` (input) and `{case_id}_map1.txt` (output)
- Runtime: ~2 minutes per case on 16-core CPU
- Total dataset generation: ~21 hours for 631 cases

**Data Format:**
- Input (`map0`): 160×80 matrix with values in [0,1] representing initial damage field
- Output (`map1`): 160×80 matrix with values in [0,1] representing final crack pattern
- Text format: space-separated values, one row per line

### Step 2: Train the Transformer Model
```bash
cd ML_code/train

# Train on 631 cases (takes ~2 hours on GPU)
python train.py \
  --data_dir ../../FEM-generated_training_datasets/631cases/ \
  --output_dir ../../Trained_Model/transformer_model_631cases/ \
  --config config.yaml \
  --gpu 0

# Train on 7176 cases (takes ~48 hours on GPU)
python train.py \
  --data_dir ../../FEM-generated_training_datasets/7176cases/ \
  --output_dir ../../Trained_Model/transformer_model_7176cases/ \
  --config config.yaml \
  --gpu 0
```

**Expected Training Output:**
```
Epoch 1/100: loss=0.234, val_loss=0.198, accuracy=0.876
Epoch 50/100: loss=0.045, val_loss=0.052, accuracy=0.962
Epoch 100/100: loss=0.023, val_loss=0.031, accuracy=0.981
Model saved to: ../../Trained_Model/transformer_model_631cases/
```

**Training Data Split:**
- Training: 80% (505 cases for 631-case dataset)
- Validation: 10% (63 cases)
- Testing: 10% (63 cases)
- Split performed with `train_test_split(random_state=42)`

### Step 3: Predict Crack Paths
```bash
cd ML_code/predict

# Single prediction
python predict.py \
  --model ../../Trained_Model/transformer_model_631cases/ \
  --input ../../porous_medium/ML/test_case_1.txt \
  --output predictions/test_case_1_predicted.txt

# Batch prediction on HPC cluster
qsub prepare.qsub  # Submits predictions for all test cases
```

**Expected Prediction Output:**
- Predicted crack path as 160×80 matrix
- Visualization saved as PNG
- Runtime: <1 second per case on GPU

### Step 4: Validate Results
```bash
cd ML_code/predict

# Compare ML predictions with FEM ground truth
python validate.py \
  --predictions predictions/ \
  --ground_truth ../../FEM-generated_training_datasets/631cases/ \
  --metrics_output validation_metrics.csv

# Generate Figure 6 from the paper
python plot_comparison.py --output paper_figures/
```

**Expected Validation Metrics:**
- IoU (Intersection over Union): >0.92
- Pixel-wise accuracy: >0.94
- Hausdorff distance: <2.5 pixels

---

## Mapping Code to Paper Results

| Paper Section | Figure/Table | Script | Expected Output |
|--------------|--------------|--------|-----------------|
| Section 3.1 | Figure 2 | `FE_code/generate_figure2.py` | J-integral validation plot |
| Section 3.2 | Figure 3 | `FE_code/generate_figure3.py` | VSBC displacement and energy |
| Section 4.1 | Figure 6 | `FE_code/generate_figure6.py` | Toughness heterogeneity |
| Section 4.2 | Figure 9 | `FE_code/generate_figure9.py` | Elastic heterogeneity |
| Section 5 | Figure 14 | `FE_code/experimental_comparison.py` | Experimental validation |
| Section 6 (ML) | Figure 15 | `ML_code/predict/plot_ml_results.py` | ML prediction accuracy |
| Section 6 (ML) | Table 2 | `ML_code/predict/validate.py` | Performance metrics |

Each script is self-contained and includes comments mapping parameters to paper equations.

---

## Phase-Field Model Parameters

All parameters from Equations (1)-(3) in the paper:

| Parameter | Symbol | Value | Location in Code |
|-----------|--------|-------|------------------|
| Fracture toughness (matrix) | G_c | 0.1 | `dp.yaml:cs0001:fracturetoughness` |
| Fracture toughness (pores) | G_c | 100.0 | `dp.yaml:cs0002:fracturetoughness` |
| Internal length scale | l_0 | 2.0 | `dp.yaml:cs0001:internalLength` |
| Young's modulus (matrix) | E | 1.0 | `dp.yaml:cs0001:YoungsModulus` |
| Poisson's ratio (matrix) | ν | 0.3 | `dp.yaml:cs0001:vsd_poissonratio` |
| Residual stiffness | η | 1e-6 | `dp.yaml:cs0001:residualstiffness` |
| VSBC exponent | n | 8 | Implicit in modulus profile |
| VSBC parameters | a, b | 0.4, 0.005 | `dp.yaml:cs0002-cs0051:YoungsModulus` |

The spatially-varying Young's modulus following E_vsd(x) = a(1-x/L_x)^n + b (Eq. 3) is implemented through the discrete values in `cs0002` through `cs0051`.

---

## Troubleshooting

**Issue:** MEF90 fails to converge  
**Solution:** Reduce time step in `dp.yaml`: change `numstep: 100` to `numstep: 200`

**Issue:** GPU out of memory during training  
**Solution:** Reduce batch size in `ML_code/train/config.yaml`

**Issue:** Different results than paper  
**Solution:** Verify all random seeds are set correctly (check `RANDOM_SEED=42` in all scripts)

---

## Citation

If you use this code, please cite:
```bibtex
@article{hao2026variable,
  title={Variable stiffness boundary condition to determine effective toughness of heterogeneous materials},
  author={Hao, Tengyuan and Piel, Adrian and Hossain, Zubaer},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={448},
  pages={118414},
  year={2026}
}
```

---

## Contact

For questions about reproducibility:
- Open an issue on GitHub
- Email: thao39@gatech.edu
```
