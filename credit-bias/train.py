# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

input_filepath = "credit-bias/data/train.csv"
output_filepath = "credit-bias/model.joblib"

# Load the processed dataset
print("Loading processed dataset")
_df = pd.read_csv(input_filepath)

# Convert boolean columns to categorical
bool_cols = _df.select_dtypes(include="bool").columns
_df[bool_cols] = _df[bool_cols].astype("category")

if not _df.empty:
    X_df = _df.drop("PaidLoan", axis=1)
    y_df = _df["PaidLoan"]

    # Splitting the dataset
    train_x, _, train_y, _ = train_test_split(
        X_df, y_df, test_size=0.25, random_state=42
    )

    # Setup the pipeline with imputation and random forest classifier
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Parameter grid for Random Forest
    param_test = {
        "classifier__max_depth": [2, 4, 8],
        "classifier__n_estimators": [10, 20, 50],
        "classifier__min_samples_split": [2, 4, 8],
    }

    # Setup the grid search
    gsearch = GridSearchCV(
        estimator=pipeline,
        param_grid=param_test,
        scoring="roc_auc",
        n_jobs=-1,
        cv=8,
        verbose=10,
    )

    # Fit the model
    gsearch.fit(train_x, train_y)

    # Logging best parameters and score
    best_params = gsearch.best_params_
    best_score = gsearch.best_score_
    print(f"Best Parameters: {best_params} | Best AUC: {best_score}")

    # Building the model with the best parameters
    rf_model = RandomForestClassifier(
        max_depth=best_params["classifier__max_depth"],
        n_estimators=best_params["classifier__n_estimators"],
        min_samples_split=best_params["classifier__min_samples_split"],
        random_state=42,
    )

    # Re-create the pipeline with the best parameters
    best_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("classifier", rf_model)]
    )

    # Refit the best model on the entire training set
    best_pipeline.fit(train_x, train_y)

    # Model serialization
    joblib.dump(best_pipeline, output_filepath)
    print("Model saved to", output_filepath)
