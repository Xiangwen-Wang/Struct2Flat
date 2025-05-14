# Struct2Flat

This repository contains the code, datasets, and candidate results for the structure-informed discovery of flat bands in two-dimensional (2D) materials. Our framework integrates a physically motivated flatness score, combination of band dispersion and density of states, with a multi-modal deep learning model trained on atomic structures and compositional descriptors. Without requiring electronic band structure inputs, the model enables scalable and interpretable prediction of flat-band candidates across large 2D material databases.

## Features
Physically motivated flatness score combining band dispersion and DOS sharpness
Multi-modal model using graph-based encoders and text-based descriptors
Fully structure-only prediction — no DFT band structure required at inference time
Applied to predict 10,000+ materials from C2DB
Includes DFT validation and topological analysis for selected candidates


## Repository Structure

Struct2Flat/
│
├── data/                # Processed and labeled datasets
├── model/               # Deep learning architecture
├── scripts/             # Flatness score computation, kagome-like and atom projection screening
├── results/             # Candidate summaries, band structures, and plots
├── requirements.txt     # Environment of the deep learning model   
└── README.md

## Data

This repository includes only the processed datasets with algorithm-labeled flatness scores used for model training and inference.

The original raw data can be accessed from the official databases:

2DMatPedia: http://www.2dmatpedia.org

C2DB: https://cmr.fysik.dtu.dk/c2db/c2db.html

## Getting Started

1. Clone the repository

2. git clone https://github.com/your-username/Struct2Flat.git

```cd Struct2Flat```

3. Set up the environment
   
```pip install -r requirements.txt```

4. Train the model

All parameters and file paths can be configured in configs.py.

```cd model```

```python train.py```

5. Predict with the model

```python prediction.py```

We provided a trained model in results folder.

6. Run inference for c2db

```python predict_c2db.py```

## Results

Over 10,000 2D materials screened

748 materials with predicted flatness scores > 0.95

Several top candidates validated with DFT and topological quantum chemistry

Results and figures from the paper are available under /results/.

## Citation
