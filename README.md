# Deep Learning-Based Modeling and Forecasting of Greenhouse Gas Emissions

Deep learning models for forecasting industrial greenhouse gas emissions using facility-level data from the U.S. EPA Greenhouse Gas Reporting Program (GHGRP), with a pipeline designed for future adaptation to Ethiopian industrial data.

## Overview

Ethiopia is rapidly industrializing, but facility-level greenhouse gas emissions data and forecasting tools remain limited. This project investigates whether deep learning can forecast industrial emissions using the detailed U.S. EPA GHGRP dataset as a foundation for future adaptation to Ethiopian facility-level data.

Three models were developed and compared:

* **LSTM**
* **GRU**
* **Hybrid LSTM-GRU**

The **GRU model achieved the best overall performance** and was integrated into a working prediction pipeline that accepts historical facility-level emission records and produces a next-year forecast.

## Dataset

**Source:** U.S. EPA Greenhouse Gas Reporting Program (GHGRP), 2011–2023.

The dataset contains facility, industry, unit, capacity, and emissions information, including:

* Facility and location information
* Industry and NAICS classification
* Reporting year
* Unit type and heat-input capacity
* CO₂, CH₄, N₂O, and biogenic CO₂

The target variable, **Total CO₂e**, combines the reported greenhouse gases using:

```text
Total CO₂e = CO₂ + Biogenic CO₂ + 25 × CH₄ + 298 × N₂O
```

## Methodology

The forecasting pipeline includes:

1. Data cleaning and missing-value handling
2. Feature engineering
3. Yeo-Johnson transformation for skewed numerical variables
4. Min-Max scaling
5. Categorical feature encoding
6. Six-year sliding-window sequence generation
7. Time-based train/validation/test splitting
8. LSTM, GRU, and Hybrid LSTM-GRU training
9. Model evaluation and forecasting

The models use separate branches for **dynamic time-series features** and **static facility characteristics**, which are combined before the final prediction layer.

For detailed preprocessing, model architectures, experiments, and statistical analysis, see the full research paper.

## Results

| Model           |        MAE |       RMSE |         R² | Parameters |
| --------------- | ---------: | ---------: | ---------: | ---------: |
| **GRU**         | **0.0265** | **0.0506** | **0.9369** | **23,537** |
| LSTM            |     0.0280 |     0.0524 |     0.9322 |     30,705 |
| Hybrid LSTM-GRU |     0.0525 |     0.0670 |     0.8892 |     55,697 |

The **GRU model performed best**, achieving the highest R² and lowest MAE and RMSE while using the fewest parameters.

A Diebold-Mariano test indicated a statistically significant difference between the GRU and LSTM forecasts (**p < 0.05**).

## Repository Structure


```text
Green-house-effect/
├── Artifact/
│   ├── feature_config.json
│   ├── minmax_scaler.joblib
│   ├── vocabularies.json
│   └── yeojohnson_lambdas.json
│
├── Code/
│   ├── Deep final.ipynb
│   └── predict_ghg_input_corrected_again_with backpadding.py
│
├── Emssion models/
│   ├── GRU_Best.keras
│   ├── HYBRID_model.keras
│   └── LSTM_best.keras
│
├── Sequence/
│   └── sequence_data (1).npz
│
├── .gitignore
├── Deep paper.pdf
└── README.md
```

## Getting Started

### Requirements

* Python 3.11
* TensorFlow
* NumPy
* Pandas
* scikit-learn
* joblib
* pyxlsb

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Run a Forecast

Run the prediction script:

```bash
python src/predict.py
```

The prediction pipeline:

1. Loads the trained GRU model.
2. Loads the saved preprocessing artifacts.
3. Prompts for a facility-level CSV file.
4. Applies the same preprocessing used during training.
5. Handles histories shorter than six years using back-padding.
6. Produces a next-year CO₂e forecast.

## Future Work

* Retrain and validate the models using Ethiopian facility-level emissions data when such data becomes available.
* Incorporate additional economic and industrial variables.
* Improve forecasting of extreme emission events.
* Develop a web-based interface for researchers, industries, and environmental agencies.
