# 📘 Used Car Price Prediction
![Python](https://img.shields.io/badge/python-3.12-blue)
![CV](https://img.shields.io/badge/CV-0.845-blue.svg)

An end-to-end machine learning project for predicting the price of used cars based on a large dataset of over 122,000 entries.
The project includes data preprocessing, EDA, feature engineering, model training, evaluation, and comparisons of multiple regression models.

## 🚀 Project Overview

This repository builds a complete ML pipeline capable of predicting used car prices using structured data.
It focuses on:

- Cleaning and preprocessing raw data
- Handling missing values
- Encoding categorical variables
- Feature engineering
- Training multiple regression models
- Evaluating model performance
- Comparing algorithms to determine the best performer
  
## 📂 Dataset

The dataset is: [USA_Cars_sales.csv (122k+rows)](https://www.kaggle.com/datasets/olivia05144/usa-used-cars-dataset)
It typically includes features such as:
- Make / Model
- Year
- Mileage
- Trim
- Engine etc....

## 🧠 Models Implemented

The project compares two regression algorithms:

- Linear Regression
- Random Forest Regressor

Metrics used:

- RMSE
- R² Score

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

## 📊 Exploratory Data Analysis (EDA)

The project includes visualizations such as:
- Distribution of car prices
- Correlation heatmaps
- Mileage vs. price
- Age vs. price
- Categorical distributions (fuel type, transmission, etc.)

## 🧹 Preprocessing Pipeline

- Handling missing values
- Removing outliers
- Encoding categorical columns
- Scaling numeric features
- Creating train/validation/test splits

## 📁 Project Structure
```
.
├── data/
│   └── Used_Cars_sales.csv
├── src/
│   ├── functions/
│   │  ├── preprocessing.py
│   │  ├── fengineering.py
│   │  ├── modeling.py
│   ├── loader/
│   │  └── load_data.py
├── main.py
└── README.md
```

## ▶️ How to Run
```bash
Clone the repository:
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

Install dependencies:
pip install -r requirements.txt

Run the main pipeline:
python main.py
```
## 📈 Results

The project prints:

- Training/Validation/Test scores
- RMSE, R²
- Confusion-matrix-style plots for regression errors
- Model comparison table
  
## ✔ Future Improvements

- Hyperparameter tuning with Optuna or GridSearch
- Deployment using FastAPI or Flask
- Streamlit dashboard for predictions
- Improved feature engineering
- Model stacking/ensembling

## 🤝 Contributions

Pull requests and suggestions are welcome!
For major changes, please open an issue first.

## 📜 License

This project is licensed under the MIT License.
