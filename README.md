# RT-Analyzer

This project revolves around the implementation of predictive models capable of determining the RT of analytes based on various chemical features.

This repository contains a set of scripts for predicting Retention Time (RT) in chromatography data. The scripts are designed to handle data preprocessing, model training, and submission of the prediction.

## Installation

To install the RT-Analyzer project, follow these steps:

1. Clone the repository: 

```bash
git clone https://github.com/your-username/RT-Analyzer.git
```


2. Install the following dependencies :

- `pandas`
- `numpy`
- `scikit-learn`
- `keras`
- `pytorch`
- `skorch`
- `xgboost`
- `rdkit`
- `matplotlib`


By running:

```bash
pip install numpy pandas scikit-learn keras pytorch skorch xgboost rdkit matplotlib
```

## Usage : Kaggle submission

To obtain the predictions submitted on Kaggle, run the following file :

```bash
python main.py
```

## Main Script Parameters

The `main.py` script features the following parameters:

- `model paths`: Path to the submissions files of individual models ('nn.csv', 'keras.csv', 'gb.csv).
- `output path`: Path to store the submission file (default: 'submission.csv').
- `submit`: Flag to indicate whether to generate Kaggle submission 
- `plot`: Flag to enable/disable plotting during model training 

## Testing Individual Models

To test individual models implemented in this project, follow these steps:

1. Open the relevant script file (e.g., `models.py`, `non_linear_models.py`, `linear_models.py`).

2. Uncomment the code section pertaining to the model of interest.

3. Adjust as needed the model testing parameters

```bash
# Model testing parameters

submit = False
plot = True
cddd = False
file_path = 'nn.csv'
```

4. Run the modified script to evaluate the selected model.

## Contribution

[@melinacherchali](https://github.com/melinacherchali)   [@slne18](https://github.com/slne18)
