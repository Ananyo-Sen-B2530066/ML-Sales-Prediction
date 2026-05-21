# ML Sales Prediction

Machine Learning based sales prediction project using video game sales data.  
The project performs exploratory data analysis, feature engineering, regression modeling, evaluation, and visualization to predict global sales.

---

## Project Structure

```text
ML-Sales-Prediction/
│
├── game_data_eda.py
├── game_regression_original.py
├── game_regression_all.py
├── vgsales.csv
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Features

- Exploratory Data Analysis (EDA)
- Missing value handling
- Feature engineering
- Target encoding
- Regression modeling
- Model comparison
- Residual analysis
- Feature importance visualization

---

## Machine Learning Models Used

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Category Encoders

---

## Dataset

Dataset used:
- `vgsales.csv`

The dataset contains:
- Game title
- Platform
- Genre
- Publisher
- Release year
- Regional sales
- Global sales

---

## Installation

Clone repository:

```bash
git clone https://github.com/YOUR_USERNAME/ML-Sales-Prediction.git
cd ML-Sales-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run

### Exploratory Data Analysis

```bash
python game_data_eda.py
```

### Original Regression Pipeline

```bash
python game_regression_original.py
```

### Enhanced Regression Pipeline

```bash
python game_regression_all.py
```

---

## Outputs

The project generates:

- Sales prediction metrics
- Regression evaluation scores
- Residual distribution plots
- Feature importance plots
- Correlation heatmaps
- Comparative model performance visualizations

---

## Author

Ananyo Sen