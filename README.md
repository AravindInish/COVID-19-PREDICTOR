# 🦠 COVID-19 Daily Cases Forecasting Using Deep Learning

<p align="center">

<img src="https://img.shields.io/badge/Deep%20Learning-LSTM-blue?style=for-the-badge">
<img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow">
<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/Time%20Series-Forecasting-purple?style=for-the-badge">
<img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">

</p>

<p align="center">
  <b>📈 A Deep Learning based time-series forecasting model for predicting daily global COVID-19 confirmed cases.</b>
</p>

---

## 🌍 Overview

The COVID-19 pandemic generated large amounts of time-series data containing daily confirmed cases, deaths, recoveries, and active cases across countries and regions.

This project applies **Long Short-Term Memory (LSTM)** networks to historical COVID-19 data to learn temporal patterns and predict the number of **daily confirmed cases**.

The model uses the previous **7 days** of COVID-19 statistics to forecast the following day's confirmed cases.

---

## 🎯 Project Objective

The primary objective is to build a deep learning model capable of learning temporal relationships in COVID-19 statistics and forecasting future daily confirmed cases.

### Key Goals

* 📊 Analyze COVID-19 time-series data
* 🧹 Clean and preprocess the dataset
* 📅 Aggregate worldwide daily statistics
* 🔄 Transform the data into sequential training samples
* 📏 Normalize numerical features
* 🧠 Train an LSTM-based deep learning model
* 📈 Predict future daily confirmed cases
* 📉 Evaluate predictions using RMSE
* 🔍 Compare actual and predicted values

---

# 🧠 How the Model Works

The model follows a time-series forecasting pipeline:

```text
              🌍 COVID-19 Dataset
                       │
                       ▼
              🧹 Data Cleaning
                       │
                       ▼
              📅 Date Processing
                       │
                       ▼
          🌎 Global Daily Aggregation
                       │
                       ▼
              📊 Feature Selection
                       │
                       ▼
             📏 MinMax Normalization
                       │
                       ▼
            🪟 7-Day Sliding Window
                       │
                       ▼
             ✂️ Train / Test Split
                       │
                       ▼
              🧠 LSTM Network
                       │
              ┌────────┴────────┐
              ▼                 ▼
         LSTM 50 Units      Dropout 20%
              │
              ▼
         LSTM 50 Units
              │
              ▼
         Dropout 20%
              │
              ▼
          Dense Output
              │
              ▼
       📈 Daily Case Prediction
              │
              ▼
          📉 RMSE Evaluation
```

---

# 📂 Dataset

The project uses the **Corona Virus Report** dataset downloaded through KaggleHub.

Dataset source:

`imdevskp/corona-virus-report`

The downloaded dataset contains:

* `country_wise_latest.csv`
* `covid_19_clean_complete.csv`
* `full_grouped.csv`
* `day_wise.csv`
* `usa_county_wise.csv`
* `worldometer_data.csv`

The project primarily uses:

`covid_19_clean_complete.csv`

The dataset contains **49,068 records and 10 columns**, including location, date, confirmed cases, deaths, recovered cases, active cases, and WHO region.

---

# 🧹 Data Preprocessing

The preprocessing pipeline performs several operations before training.

### 1️⃣ Date Conversion

The `Date` column is converted into a datetime format to enable time-series operations.

### 2️⃣ Duplicate Detection

Duplicate rows are checked before model preparation.

The notebook reports:

```text
Duplicate rows: 0
```

### 3️⃣ Missing Value Handling

Missing `Province/State` values are replaced with:

```text
Unknown
```

After preprocessing, the dataset contains no missing values in the processed columns.

### 4️⃣ Data Correction

The dataset contained **18 negative Active case values**.

These values were corrected to `0` during preprocessing.

---

# 🌎 Global Time-Series Preparation

The data is aggregated by date to obtain global daily statistics.

The forecasting dataset uses three features:

```text
Daily_Confirmed
Daily_Deaths
Daily_Recovered
```

These features are then normalized using:

```text
MinMaxScaler(feature_range=(0, 1))
```

---

# 🪟 Sequence Generation

The model uses a **7-day look-back window**.

In simple terms:

```text
Previous 7 Days
      │
      ▼
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ D-7 │ D-6 │ D-5 │ D-4 │ D-3 │ D-2 │ D-1 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                         │
                         ▼
                 🧠 LSTM Model
                         │
                         ▼
                    Predict D
```

Each sequence contains:

**7 time steps × 3 features**

The target is the next day's `Daily_Confirmed` value.

---

# 🧠 Deep Learning Architecture

The project uses a **stacked LSTM architecture** implemented using TensorFlow/Keras.

### Architecture

```text
Input
  │
  │  7 Timesteps × 3 Features
  ▼
┌───────────────────────────┐
│      LSTM - 50 Units      │
│    return_sequences=True  │
└─────────────┬─────────────┘
              │
              ▼
        Dropout - 20%
              │
              ▼
┌───────────────────────────┐
│      LSTM - 50 Units      │
│    return_sequences=False │
└─────────────┬─────────────┘
              │
              ▼
        Dropout - 20%
              │
              ▼
┌───────────────────────────┐
│       Dense - 1 Unit      │
└─────────────┬─────────────┘
              │
              ▼
📈 Predicted Daily Confirmed Cases
```

The implemented network contains:

* 🧠 LSTM layer with 50 units
* 🛡️ Dropout layer with 20%
* 🧠 Second LSTM layer with 50 units
* 🛡️ Dropout layer with 20%
* 🎯 Dense output layer with 1 unit

---

# ⚙️ Training Configuration

The model is trained using:

| Parameter         |                 Value |
| ----------------- | --------------------: |
| Look-back window  |                7 days |
| LSTM units        |               50 + 50 |
| Dropout           |                   20% |
| Epochs            |                    50 |
| Batch size        |                    32 |
| Validation split  |                   10% |
| Output            | Daily confirmed cases |
| Evaluation metric |                  RMSE |

---

# 📊 Train / Test Pipeline

```text
📊 Processed Dataset
        │
        ▼
📏 Normalize Features
        │
        ▼
🪟 Create 7-Day Sequences
        │
        ▼
┌─────────────────────┐
│   80% Training Data │
└──────────┬──────────┘
           │
           ▼
      🧠 LSTM Training
           │
           ▼
      📚 Learned Patterns
           
┌─────────────────────┐
│    20% Testing Data │
└──────────┬──────────┘
           │
           ▼
      🔮 Predictions
           │
           ▼
      📉 RMSE Score
```

---

# 📈 Model Evaluation

The model compares predicted daily confirmed cases against the actual values in the test dataset.

The notebook reports:

```text
Test RMSE: 58763.83
```

The project also generates an **Actual vs Predicted Daily Confirmed Cases** visualization to inspect how closely the model follows the observed time-series pattern.

---

# 📉 Prediction Visualization

The final evaluation compares:

```text
🔵 Actual Daily Confirmed Cases
🔴 Predicted Daily Confirmed Cases
```

Conceptually:

```text
Cases
  │
  │       🔵 Actual
  │      ╱╲       ╱╲
  │     ╱  ╲     ╱  ╲
  │  🔴╱────╲───╱────╲
  │
  └────────────────────────► Time
```

The notebook plots the actual and predicted values across the test period.

---

# 🛠️ Tech Stack

### Programming

🐍 Python

### Data Science

* 🐼 Pandas
* 🔢 NumPy
* 📊 Matplotlib
* 📈 Seaborn
* 🔄 Scikit-learn

### Deep Learning

* 🧠 TensorFlow
* 🤖 Keras
* LSTM
* Dropout
* Dense layers

### Dataset

* KaggleHub
* COVID-19 Corona Virus Report dataset

---

# 📁 Project Structure

```text
COVID-19-Forecasting/
│
├── 📓 COVID_19.ipynb
│
├── 📄 README.md
│
└── 📊 Dataset
    ├── country_wise_latest.csv
    ├── covid_19_clean_complete.csv
    ├── full_grouped.csv
    ├── day_wise.csv
    ├── usa_county_wise.csv
    └── worldometer_data.csv
```

---

# 🚀 Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/AravindInish/COVID-19-Forecasting.git
cd COVID-19-Forecasting
```

## 2️⃣ Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow kagglehub
```

## 3️⃣ Launch the Notebook

```bash
jupyter notebook COVID_19.ipynb
```

Or open the notebook directly using **Google Colab**.

---

# 🔬 Complete Workflow

```text
          🚀 START
             │
             ▼
       📥 Load Dataset
             │
             ▼
       🔍 Explore Data
             │
             ▼
       🧹 Clean Data
             │
             ▼
       📅 Process Dates
             │
             ▼
     🌎 Aggregate Globally
             │
             ▼
      📊 Select Features
             │
             ▼
       📏 Normalize Data
             │
             ▼
      🪟 Create Sequences
             │
             ▼
       ✂️ Split Dataset
             │
             ▼
       🧠 Build LSTM
             │
             ▼
       🏋️ Train Model
             │
             ▼
       🔮 Make Predictions
             │
             ▼
       📉 Calculate RMSE
             │
             ▼
     📊 Visualize Results
             │
             ▼
          🏁 END
```

---

# 💡 Why LSTM?

Traditional machine learning models can struggle to naturally represent dependencies across sequential observations.

LSTM networks are designed for sequence-based data and can learn patterns across previous time steps.

For this project:

```text
Past COVID-19 Trends
        ↓
Temporal Patterns
        ↓
LSTM Memory
        ↓
Future Case Prediction
```

This makes LSTM a suitable deep learning approach for experimenting with COVID-19 time-series forecasting.

---

# ⚠️ Limitations

This project is primarily an educational and experimental forecasting implementation.

Important limitations include:

* 📅 The dataset represents historical COVID-19 observations.
* 🌍 The model forecasts global daily confirmed cases rather than individual-country predictions.
* 🧠 Predictions depend heavily on the historical patterns available in the dataset.
* 📉 RMSE alone does not provide a complete picture of forecasting quality.
* 🔮 Real-world epidemiological forecasting requires additional factors such as interventions, variants, vaccination, mobility, testing, demographics, and public-health policy.

Therefore, the model should **not be interpreted as a medical or public-health decision system**.

---

# 🚀 Future Improvements

Potential improvements include:

* 🌎 Country-level forecasting
* 📍 Regional forecasting
* 🧠 Bidirectional LSTM
* ⚡ GRU comparison
* 🔥 Transformer-based forecasting
* 📊 Additional epidemiological features
* 🔧 Hyperparameter optimization
* 📉 MAE, MAPE and R² evaluation
* 🔮 Multi-step forecasting
* 📦 Model deployment using Streamlit
* 🌐 REST API deployment
* 📱 Interactive prediction dashboard
* 🔄 Automated data updates

---

# 👨‍💻 Author

### Aravind Inish

Student | AI & Machine Learning Enthusiast | Developer

🔗 GitHub:
https://github.com/AravindInish

---

# ⭐ Support

If this project was useful or interesting:

⭐ **Star the repository**

🍴 **Fork the project**

🐛 **Open an issue**

💡 **Suggest an improvement**

🤝 **Contribute**

---

<p align="center">

### 🦠 COVID-19 Forecasting × 🧠 Deep Learning × 📈 Time Series

<b>Turning historical data into predictive insights.</b>

</p>
