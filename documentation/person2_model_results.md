# Person 2 - Machine Learning Results

## Dataset

The final processed dataset contains 32,593 student records
and 27 columns.

The target variable is `at_risk`.

- At Risk (1): 17,208 students
- Not At Risk (0): 15,385 students

## Input Features

`id_student` was excluded because it is an identifier.

The remaining 25 features consist of:

- 8 categorical features
- 17 numerical features

## Preprocessing

Numerical features:
- Missing values: median imputation
- Scaling: StandardScaler

Categorical features:
- Missing values: most-frequent imputation
- Encoding: OneHotEncoder

## Train/Test Split

- Training: 80%
- Testing: 20%
- Random state: 42
- Stratified by `at_risk`

## Models Evaluated

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

## Best Model

Gradient Boosting was selected based on the overall model
comparison.

Accuracy: 93.53%

Precision: 96.78%

Recall: 90.76%

F1 Score: 93.67%

## Model File

The trained model is stored at:

`models/best_model.pkl`

The complete preprocessing pipeline is included in the saved model.

## Handoff to Person 3

Person 3 can load `best_model.pkl` using joblib and provide
new student feature data to the pipeline for prediction.