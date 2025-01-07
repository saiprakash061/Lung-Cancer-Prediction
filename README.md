
# Lung Cancer Prediction

## Project Overview
This project aims to develop a machine learning model to predict lung cancer risk based on a survey dataset. The dataset contains information on various factors such as smoking habits, age, gender, and other health conditions. The project implements multiple machine learning models to evaluate their predictive performance and identify the most effective approach.

---

## Dataset
- **Name**: `survey lung cancer.csv`
- **Number of Records**: 309
- **Features**:
  - **Input Features**:
    - Gender (Categorical: M/F)
    - Age (Numerical)
    - Smoking, Yellow Fingers, Anxiety, Peer Pressure, Chronic Disease, Fatigue, Allergy, Wheezing, Alcohol Consuming, Coughing, Shortness of Breath, Swallowing Difficulty, Chest Pain (Binary: 0/1)
  - **Target Feature**:
    - Lung Cancer (Binary: Yes/No)

---

## Requirements
To set up the environment and run the project, install the following libraries:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn
```

**Environment**:  
- Python 3.8 or later  
- Jupyter Notebook or any Python IDE

---

## Preprocessing Steps
1. **Data Cleaning**:
   - Verified no missing values in the dataset.
   - Encoded categorical variables:
     - Gender: `M -> 1`, `F -> 0`
     - Lung Cancer: `Yes -> 1`, `No -> 0`
2. **Data Balancing**:
   - Handled class imbalance using Random Over Sampling.
3. **Feature Scaling**:
   - Standardized numerical features using `StandardScaler`.
4. **Data Splitting**:
   - Split the dataset into training (80%) and testing (20%) subsets.

---

## Models Involved
The following machine learning models were implemented, evaluated, and compared:

1. **Linear Regression**:
   - Regression model for baseline comparison.
   - Accuracy: 64.04%

2. **Random Forest Classifier**:
   - An ensemble-based model using decision trees.
   - High accuracy: 99.07%
   - Robust to overfitting and interpretable feature importance.

3. **K-Nearest Neighbors (KNN)**:
   - A distance-based classifier.
   - Accuracy: 93.52%
   - Best for small datasets with balanced features.

4. **Decision Tree Classifier**:
   - A tree-structured model that splits data based on feature values.
   - Accuracy: 96.29%
   - Simple and interpretable but prone to overfitting.

5. **Naive Bayes Classifier**:
   - A probabilistic model assuming feature independence.
   - Accuracy: 95.70%
   - Suitable for small datasets and quick predictions.

---

## Evaluation Metrics
Metrics used to evaluate model performance include:
- **Regression Models**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score
- **Classification Models**:
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

---

## Results Summary
| Model                 | Accuracy   | Precision (Healthy/Suspected) | Recall (Healthy/Suspected) |
|-----------------------|------------|--------------------------------|----------------------------|
| Linear Regression     | 64.04%     | N/A                            | N/A                        |
| Random Forest         | 99.07%     | 0.98 / 1.00                    | 1.00 / 0.98                |
| K-Nearest Neighbors   | 93.52%     | 0.88 / 1.00                    | 1.00 / 0.88                |
| Decision Tree         | 96.29%     | 0.93 / 1.00                    | 1.00 / 0.93                |
| Naive Bayes           | 95.70%     | N/A                            | N/A                        |

---

## Instructions to Run the Project
1. Clone the repository and place the dataset in the same directory as the Jupyter Notebook.
2. Install the required libraries using the commands provided in the "Requirements" section.
3. Open the `lung-cancer-eda.ipynb` notebook.
4. Execute the cells sequentially to:
   - Preprocess the data.
   - Analyze the dataset through visualizations.
   - Train and evaluate the machine learning models.
5. Review the output for metrics, accuracy, and visualizations.

---

## Future Enhancements
- Experiment with advanced models like **XGBoost** or **Neural Networks**.
- Apply **hyperparameter tuning** for optimal model performance.
- Expand the dataset for better generalization and robustness.
- Investigate additional factors or external datasets that may impact lung cancer prediction.

---
