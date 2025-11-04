# Patient stratification using Time-Aware Multi-modal autoEncoder(TAME)

This project implements **TAME (Time-Aware Multi-modal AutoEncoder)** for patient stratification using the **NephroCAGE** dataset.  
TAME is designed to handle heterogeneous, longitudinal, and multi-modal clinical data to identify patient subgroups and risk profiles.

## Data requirement 
This project utilizes the NephroCAGE dataset, which must be placed in the NephroCAGE/ folder before running the pipelines.

## Installation and Environment Setup
Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Once the virtual environment is activated, install the dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
### TAME training pipeline
Run this for training pipeline:

```bash
python src/training.py
```
Trained model is saved in (`model_checkpoints`)

### Similarity matrix calculation
To extract the real time series values and impute them with TAME output:
```bash
python src/dist_mat.py
```

### Clustering using similarity matrix
Patient subtyping on similarity matrix and evaluation are implemented in Jupyter notebook: (`notebooks/clustering_similarity.ipynb`)

### Clustering using TAME-based embeddings
Embeddings extraction and patient subtyping on TAME-based embeddings, along with validation of subgroups are implemented in Jupyter notebook: (`notebooks/clustering_emb.ipynb`)

### Health outcomes prediction
Related embeddings extraction, training classifiers and SHAP analysis are implemented in Jupyter notebook: (`notebooks/classification.ipynb`)

### Source Code (`src/`)

- `config.py` – Global configuration parameters (e.g., list of numerical features, embedding size).
- `dist_mat.py` –  Compute distance matrices using dtw.
- `dtw.py` – Dynamic Time Warping (DTW) utilities.
- `function.py` – General helper functions (e.g., nRMSE loss).
- `myloss.py` – Custom loss function implementations.
- `preprocessing.py` – Data preprocessing pipeline.
- `tame.py` – Main TAME model implementation.
- `training.py` – TAME Training and evaluation pipeline.
- `wkmeans.py` – Functions for Weighted K-Means clustering.

### Notebooks (`notebooks/`)

- `classification.ipynb` - Experiments on patient classification using learned embeddings.
- `clustering_emb.ipynb` - Clustering performed directly on embeddings to explore patient subgroups.
- `clustering_similarity.ipynb` – Clustering based on similarity/distance matrices.
- `evaluation.ipynb` – Experiments on using raw aggregated features for classification for evaluation.
- `interpretation.ipynb` – Model interpretation and analysis (e.g., attention, reconstruction plots).
- `visualization.ipynb` – Visualization of dataset characteristics.

### Classifiers (`models/`)
This folder includes trained classifiers for graft loss, mortality and rejection.

### Trained TAME (`model_checkpoints/`)
The trained TAME on NephroCAGE is included in this folder.


## Results
Stratification experiments showed that clustering patients with TAME-based embeddings yields meaningful and valid subgroups, whereas clustering with temporal similarity matrices did not produce distinct groups. Furthermore, classification using the learned embeddings demonstrated strong performance in predicting graft loss and mortality, which altogether demonstrates that TAME is able to provide informative embeddings for this dataset.





