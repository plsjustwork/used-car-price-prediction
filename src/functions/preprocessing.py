import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def drop_unnecessary_columns(df, columns=["ID", "yearsold"]):
    for col in columns:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def missing_strategy(pct):
    if pct < 5:
        return "drop_rows"
    elif pct < 15:
        return "simple_impute"
    elif pct < 40:
        return "advanced_impute"
    else:
        return "drop_column"
    
def analyze_missing_values(df):
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    rows_to_drop_cols = []
    simple_impute_cols = []
    advanced_impute_cols = []
    columns_to_drop = []

    for col, pct in missing_percentage.items():
        strategy = missing_strategy(pct)

        if strategy == "drop_rows":
            rows_to_drop_cols.append(col)
        elif strategy == "simple_impute":
            simple_impute_cols.append(col)
        elif strategy == "advanced_impute":
            advanced_impute_cols.append(col)
        elif strategy == "drop_column":
            columns_to_drop.append(col)

    return rows_to_drop_cols, simple_impute_cols, advanced_impute_cols, columns_to_drop

def apply_missing_strategies(df, rows_to_drop_cols, simple_impute_cols, advanced_impute_cols, columns_to_drop):
    
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    if len(rows_to_drop_cols) > 0:
        df = df.dropna(subset=rows_to_drop_cols)

    if len(simple_impute_cols) > 0:
        num_cols = [col for col in simple_impute_cols if col in df.select_dtypes(include=[np.number]).columns]
        cat_cols = [col for col in simple_impute_cols if col in df.select_dtypes(include=['object']).columns]

        if len(num_cols) > 0:
            simputer_num = SimpleImputer(strategy="median")
            df[num_cols] = simputer_num.fit_transform(df[num_cols])

        if len(cat_cols) > 0:
            simputer_cat = SimpleImputer(strategy="most_frequent")
            df[cat_cols] = simputer_cat.fit_transform(df[cat_cols])

    if len(advanced_impute_cols) > 0:
        num_cols_adv = [col for col in advanced_impute_cols if col in df.select_dtypes(include=[np.number]).columns]

        if len(num_cols_adv) > 0:
            knn = KNNImputer(n_neighbors=5)
            df[num_cols_adv] = knn.fit_transform(df[num_cols_adv])

        cat_cols_adv = [col for col in advanced_impute_cols if col in df.select_dtypes(include=['object']).columns]
        if len(cat_cols_adv) > 0:
            simputer_cat_adv = SimpleImputer(strategy="most_frequent")
            df[cat_cols_adv] = simputer_cat_adv.fit_transform(df[cat_cols_adv])

    return df

def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    return df

def apply_rare_encoding(df, threshold=0.02):
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        df[col] = df[col].replace(rare, "Other")

    return df


def build_preprocessor(df):

    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num_scaler", StandardScaler(), numeric_features)
        ],
        remainder="drop"
    )

    return preprocessor

def preprocess_dataframe(df):

    df = drop_unnecessary_columns(df)
    rows_to_drop, simple_cols, advanced_cols, cols_to_drop = analyze_missing_values(df)
    df = apply_missing_strategies(df, rows_to_drop, simple_cols, advanced_cols, cols_to_drop)
    df = remove_outliers(df)
    df = apply_rare_encoding(df)

    return df