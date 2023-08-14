# Diabetes Prediction Machine Learning Model

This repository contains code for creating a machine-learning model to predict the likelihood of diabetes using a diabetes prediction dataset. The model uses a RandomForestClassifier to perform the prediction. The dataset is preprocessed and features are selected to train and test the model.

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- scikit-learn

You can install these libraries using the following command:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
```

Navigate to the project directory:
```bash
cd diabetes-prediction
```

Update the dataset path in the code:
Open the Python script diabetes_prediction_model.py and replace the file path /Users/aquachat77/Downloads/diabetes_prediction_dataset.csv with the path to your own diabetes prediction dataset. 

The dataset I used: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

Run the script:
```bash
python diabetes_prediction_model.py
```

## Model Evaluation and Feature Importance:
The script will print the accuracy of the trained model on the test dataset and the ordered list of key contributing features to the prediction.

## Saved Model:
The trained RandomForestClassifier model will be saved as diabetes_model.joblib.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.
