import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv(r"C:\Users\HP\Documents\CodeClause\Personality Detection\train dataset.csv")

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

input_cols = ['Gender', 'Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']
output_cols = ['Personality (Class label)']

scaler = StandardScaler()
data[input_cols] = scaler.fit_transform(data[input_cols])

X_train, X_test, Y_train, Y_test = train_test_split(data[input_cols], data[output_cols], test_size=0.2, random_state=42)

gb_model = GradientBoostingClassifier(random_state=42)

param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

grid_search_gb = GridSearchCV(gb_model, param_grid_gb, cv=StratifiedKFold(n_splits=5), scoring='accuracy')

grid_search_gb.fit(X_train, Y_train.values.ravel())

print("Best Parameters for Gradient Boosting:", grid_search_gb.best_params_)

best_gb_model = grid_search_gb.best_estimator_
y_pred_gb = best_gb_model.predict(X_test)

accuracy_gb = accuracy_score(Y_test, y_pred_gb)
print(f"Accuracy for Gradient Boosting: {accuracy_gb * 100:.2f}%")

joblib.dump(best_gb_model, "best_gb_model.pkl")
