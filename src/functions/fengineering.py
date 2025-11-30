from datetime import datetime
import re

def feature_engineering(df):

    if "Year" in df.columns:
        current_year = datetime.now().year
        df["car_age"] = current_year - df["Year"]
        df["car_age"] = df["car_age"].clip(lower=0)

    if "Engine" in df.columns:
        df["Engine_Liters"] = df["Engine"].str.extract(r'(\d\.\d)').astype(float)
        df["Engine_Cylinders"] = df["Engine"].str.extract(r'V(\d)').astype(float)
        df["Turbo"] = df["Engine"].str.contains("Turbo", case=False, na=False).astype(int)


    if "Mileage" in df.columns and "car_age" in df.columns:
        df["Mileage_per_year"] = df["Mileage"] / (df["car_age"].replace(0, 1))

    return df