# Mechanics-Informed-Machine-Learning-Prediction-of-Crack-Path-in-Heterogeneous-Materials
Mechanics-informed ML framework for predicting crack paths in porous media. Combines FEM data with a Transformer model using physics-guided domain extraction and VSBC for stable training. Provides fast, accurate, and generalizable alternatives to costly FEM simulations.

## Repository Structure (For detailed information, please read the read_me file for each subdirectory)

# FEM-generated_training_datasets/
Contains the FEM simulation datasets used for training the Transformer models.  
- `631cases.zip` → 631 cases with **two-pore** combinations.  
- `7176cases.zip` → 7176 cases with **three-pore** combinations.  
- Example format:
  - Input matrix: `8_map0.txt`  
  - Output matrix: `8_map1.txt`  
  - (`8` is the case number.)  

# FE_code/
Contains scripts and configuration files for running FEM simulations with MEF90.  
- `dp.yaml`: Input file for MEF90 simulations to generate the ground-truth crack path.

# ML_code/
Contains the machine learning training and prediction scripts.  
- `train.py`: Transformer-based training on spatiotemporal FEM datasets.  
- `predict.py`: Predicts crack paths in porous materials using trained ML models.  

# Trained_Model/
Contains trained Transformer models.  
- `transformer_model_631cases`: Trained model using the `631cases` dataset.
- `transformer_model_7176cases`: Trained model using the `7176cases` dataset.
  - Note: The model size exceeds GitHub’s upload limit.  
  - Download from Dropbox: [https://www.dropbox.com/scl/fi/5nqqvj2hvbeo9vjjch6ah/transformer_model_7176cases.zip?rlkey=b7lsaqd0kgw8vzmy1le36iuwl&dl=0]  

# porous_medium/
Contains FEM and ML domains of porous materials used in the simulations.  

# `LICENSE`
Open-source license information.  

# `README.md`
This file.  

---

## Running FEM Simulations with MEF90

1. Install **MEF90** (https://github.com/bourdin/mef90).  
2. Prepare the following input files inside the directory:  
   - `dp.yaml` → Input file that defines the FEM setup and parameters.  
   - Input domain file (`*.gen`).
     
## Running the Machine Learning Code

The ML workflow uses TensorFlow/Keras to train and apply a Transformer model for predicting crack paths from FEM-generated datasets.

1. Run `train.py` with the datasets provided in the `FEM-generated_training_datasets` directory to train the machine learning model.
2. Use the trained model with `predict.py` to predict crack paths for porous domains located in the `porous_medium/ML` directory.
   


