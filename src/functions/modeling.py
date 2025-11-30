import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
    n_estimators=200,
    max_depth=30,
    min_samples_split=3,
    n_jobs=-1,
    random_state=42
)

    model.fit(X_train, y_train)
    return model


def evaluate_regression(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    return {
        "MSE": mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }