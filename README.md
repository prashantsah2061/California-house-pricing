# 🏠 California Housing Price Prediction

This project uses machine learning (Scikit-learn) to predict housing prices in California based on various features such as location, median income, and proximity to the ocean.
---


## 📁 Project Structure

california-house-pricing/
├── main.py # Main script to train model and do inference
├── housing.csv # Dataset file 
├── input.csv # Input for inference (auto-generated)
├── output.csv # Output predictions (auto-generated)
├── model.pkl # Trained RandomForest model (auto-generated)
├── pipeline.pkl # Data preprocessing pipeline (auto-generated)
├── requirements.txt # Required Python packages
└── README.md # You're reading it!



---

## ⚙️ Features

- Stratified sampling based on income category
- Numerical + Categorical preprocessing pipelines
- Trained using `RandomForestRegressor`
- Input-output pipeline with `.csv` format
- Automatically saves model and prediction pipeline

---

## 🧠 ML Techniques Used

- **StratifiedShuffleSplit** for robust training/test split
- **OneHotEncoding** for categorical variables
- **StandardScaler** + **SimpleImputer** for numerical features
- **RandomForestRegressor** as the main model
- **Joblib** for model/pipeline serialization

---

## 🚀 How to Run

### 1. Clone the project or download the files.

### 2. Install dependencies

```bash
pip install -r requirements.txt
