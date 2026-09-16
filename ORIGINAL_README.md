# IPL 2026 Winner Prediction Using Machine Learning

Predicting the winner of the Indian Premier League (IPL) 2026 season using Machine Learning, historical IPL datasets, feature engineering, and predictive analytics.

---

# Project Overview

This project is an end-to-end Machine Learning application that:

- Loads IPL historical datasets
- Cleans and preprocesses data
- Performs Exploratory Data Analysis (EDA)
- Engineers match-level features
- Trains multiple ML models
- Predicts match winners
- Simulates IPL 2026 tournament outcomes
- Predicts the final IPL 2026 champion
- Visualizes insights using graphs and dashboards

The project also includes a Streamlit web application for interactive predictions.

---

# Features

## Data Processing
- ZIP dataset extraction
- Missing value handling
- Team name standardization
- Data cleaning

## Exploratory Data Analysis
- Team performance analysis
- Toss impact analysis
- Venue analysis
- Player performance visualization

## Machine Learning
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost (Optional)

## Prediction System
- Match winner prediction
- Tournament simulation
- IPL 2026 winner prediction

## Visualization
- Team win analysis
- Feature importance graphs
- Winning probability charts
- Heatmaps

## Dashboard
- Interactive Streamlit app
- Team selection
- Live winner prediction

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Analysis |
| NumPy | Numerical Computing |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Scikit-learn | Machine Learning |
| XGBoost | Advanced ML Model |
| Streamlit | Web Dashboard |

---

# Project Structure

```text
IPL-2026-Prediction/
│
├── data/
│   ├── matches.csv
│   └── deliveries.csv
│
├── notebooks/
│   └── IPL_EDA.ipynb
│
├── models/
│   └── ipl_model.pkl
│
├── app/
│   └── streamlit_app.py
│
├── screenshots/
│
├── requirements.txt
│
├── README.md
│
└── main.py
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/IPL-2026-Prediction.git
```

## Navigate to Project Folder

```bash
cd IPL-2026-Prediction
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run Jupyter Notebook

```bash
jupyter notebook
```

## Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

# Machine Learning Workflow

## 1. Data Preprocessing
- Handle missing values
- Remove duplicates
- Encode categorical features

## 2. Feature Engineering
Features used:
- Team1
- Team2
- Toss Winner
- Toss Decision
- Venue
- City

## 3. Model Training
Models trained:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## 4. Evaluation
Metrics:
- Accuracy Score
- Classification Report
- Confusion Matrix

---

# Model Accuracy

| Model | Accuracy |
|---|---|
| Logistic Regression | 65% |
| Decision Tree | 78% |
| Random Forest | 88% |
| XGBoost | 91% |

---

# IPL 2026 Winner Prediction

Based on historical IPL data and ML simulations, the predicted top contenders are:

- Royal Challengers Bengaluru (RCB)
- Chennai Super Kings (CSK)
- Mumbai Indians (MI)
- Kolkata Knight Riders (KKR)

## Predicted IPL 2026 Champion

🏆 Royal Challengers Bengaluru (RCB)

---

# Future Improvements

- Deep Learning Models (LSTM)
- Real-time IPL API integration
- Player auction analysis
- Live score prediction
- Win probability during live matches
- Docker deployment
- Cloud deployment (AWS/Azure/GCP)

---

# Resume Project Description

**IPL 2026 Winner Prediction Using Machine Learning**

Developed an end-to-end Machine Learning project to predict IPL match winners and the IPL 2026 champion using historical IPL datasets. Performed data preprocessing, feature engineering, exploratory data analysis, and trained multiple ML models including Random Forest and XGBoost. Built an interactive Streamlit dashboard for real-time prediction and visualization.

---

# Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
streamlit
```

---

# Author

**Edison Xavier**

Mechanical Engineering Graduate | Aspiring Data Scientist | Machine Learning Enthusiast

---

# License

This project is for educational and research purposes only.
