import pandas as pd

from src.loader.load_data import load_dataset
from src.functions.preprocessing import preprocess_dataframe, build_preprocessor
from src.functions.modeling import train_linear_regression, train_random_forest, evaluate_regression
from src.functions.fengineering import feature_engineering
from sklearn.model_selection import train_test_split


def main():

    print("\n🔍 Loading dataset...")
    df = load_dataset("data/used_car_sales.csv")
    print(f"Dataset loaded with shape: {df.shape}")

    df = feature_engineering(df)

    print("\n🧹 Starting preprocessing...")
    df_clean = preprocess_dataframe(df)
    print(f"After preprocessing shape: {df_clean.shape}")

    print("\n📊 Splitting into train/validation/test...")

    X = df_clean.drop(columns=["pricesold"])
    y = df_clean["pricesold"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42
    )

    print(f"Train size: {X_train.shape[0]}")
    print(f"Validation size: {X_val.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed   = preprocessor.transform(X_val)
    X_test_processed  = preprocessor.transform(X_test)

    print("Number of features after encoding:", X_train_processed.shape[1])

    print("\n🤖 Training Linear Regression model...")
    lr_model = train_linear_regression(X_train_processed, y_train)

    print("\n🌲 Training Random Forest model...")
    rf_model = train_random_forest(X_train_processed, y_train)

    print("\n📉 Validation Results:")
    lr_val_results = evaluate_regression(lr_model, X_val_processed, y_val)
    rf_val_results = evaluate_regression(rf_model, X_val_processed, y_val)

    print("------------------------------------")
    print("Linear Regression (Validation):")
    print(f"  MSE: {lr_val_results['MSE']:.2f}")
    print(f"  R² : {lr_val_results['R2']:.4f}")

    print("\nRandom Forest (Validation):")
    print(f"  MSE: {rf_val_results['MSE']:.2f}")
    print(f"  R² : {rf_val_results['R2']:.4f}")

    print("\n📈 Final Test Results:")
    lr_test_results = evaluate_regression(lr_model, X_test_processed, y_test)
    rf_test_results = evaluate_regression(rf_model, X_test_processed, y_test)

    print("------------------------------------")
    print("Linear Regression (Test):")
    print(f"  MSE: {lr_test_results['MSE']:.2f}")
    print(f"  R² : {lr_test_results['R2']:.4f}")

    print("\nRandom Forest (Test):")
    print(f"  MSE: {rf_test_results['MSE']:.2f}")
    print(f"  R² : {rf_test_results['R2']:.4f}")

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
