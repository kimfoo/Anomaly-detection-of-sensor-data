# Isolation Forest Anomaly Detection

## Overview
This project implements anomaly detection using the Isolation Forest algorithm. It identifies outliers in a dataset that represent anomalies. The model is tuned using hyperparameter optimization to achieve the best F1 score.

## Features
- Anomaly detection using Isolation Forest
- Hyperparameter tuning with GridSearchCV
- Evaluation metrics: Precision, Recall, and F1-score

## Installation

### Prerequisites
- Python 3.6+
- Pandas
- Scikit-learn

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation
Ensure your dataset is in CSV format with a column named `Anomaly` indicating anomalies (-1 for anomaly, 1 for normal).

### Running the Model
1. Load your dataset:
    ```python
    import pandas as pd

    encoded_data = pd.read_csv('path_to_your_encoded_data.csv')
    ```

2. Perform hyperparameter tuning and anomaly detection:
    ```python
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer, f1_score

    # Prepare the data
    feature_names = encoded_data.drop(columns=['Anomaly']).columns
    features = encoded_data[feature_names]
    labels = encoded_data['Anomaly']

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_samples': ['auto', 0.5, 0.75],
        'contamination': [0.01, 0.03, 0.05, 0.1],
        'max_features': [1.0, 0.8, 0.5],
        'bootstrap': [True, False],
    }

    # Custom scorer using F1 score
    f1_scorer = make_scorer(f1_score, pos_label=-1)

    # Initialize and fit GridSearchCV
    iso_forest = IsolationForest(random_state=42)
    grid_search = GridSearchCV(estimator=iso_forest, param_grid=param_grid, scoring=f1_scorer, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(features, labels)

    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best F1 Score:", best_score)
    ```

## Evaluation Metrics
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **F1-Score**: The harmonic mean of precision and recall.

## Example Output
An example of the output after running the model:
```bash
Best Parameters: {'bootstrap': False, 'contamination': 0.05, 'max_features': 0.8, 'max_samples': 'auto', 'n_estimators': 100}
Best F1 Score: 0.75
