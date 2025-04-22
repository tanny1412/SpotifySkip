<p align="center">
  <img src="image.png" alt="Spotify Skip Predictor" width="300">
</p>
<h1 align="center">🎵 Spotify Skip Predictor 🚀</h1>
<p align="center">Predict whether a Spotify track will be skipped using Machine Learning 🤖</p>

[![Python Version](https://img.shields.io/badge/python-%3E%3D3.8-blue)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-locked-green?logo=pip)](requirements.txt)

---

## 🌟 Table of Contents
- [🚀 Introduction](#-introduction)
- [📊 Dataset](#-dataset)
- [🛠 Features](#-features)
- [⚙️ Installation](#-installation)
- [🎯 Usage](#-usage)
  - [🤖 Training & Saving Model](#-training--saving-model)
  - [🔬 Running the Evaluation Pipeline](#-running-the-evaluation-pipeline)
  - [🌐 Starting the API Server](#-starting-the-api-server)
  - [🖼️ Launching the Streamlit UI](#-launching-the-streamlit-ui)
- [📂 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [🙏 Acknowledgements](#-acknowledgements)

---

## 🚀 Introduction

Spotify Skip Predictor is a predictive analytics project that leverages your streaming history to forecast whether a user will skip a track. It features:

- **Data preprocessing** with temporal and contextual feature engineering.
- **Imbalanced data handling** using SMOTE.
- **Model training & hyperparameter tuning** for Random Forest and XGBoost.
- **RESTful API** built with FastAPI for real-time predictions.
- **Interactive UI** via Streamlit for seamless end-user experience.

---

## 📊 Dataset

We use the [Shopify Streaming History Dataset](https://www.kaggle.com/arshmankhalid/shopify-streaming-history-dataset) from Kaggle, which includes:

- `spotify_history.csv`: User listening sessions with playback details.
- `spotify_data_dictionary.csv`: Descriptions of each field.

Ensure both files are placed in the project root.

---

## 🛠 Features

| Feature         | Type       | Description                                         |
|-----------------|------------|-----------------------------------------------------|
| `hour`          | Numeric    | Hour of day when playback started (0–23)            |
| `month`         | Numeric    | Month of year (1–12)                                |
| `weekday`       | Numeric    | Day of week (0=Mon … 6=Sun)                         |
| `platform`      | Categorical| Encoded platform (web, iOS, Android, desktop, …)    |
| `reason_start`  | Categorical| Encoded reason playback started                     |
| `shuffle`       | Boolean    | Shuffle mode flag (0 = off, 1 = on)                 |
| `skipped`       | Target     | 0 = played through, 1 = skipped                     |

---

## ⚙️ Installation

```bash
# Clone repo
git clone https://github.com/your-username/spotify-skip-predictor.git
cd spotify-skip-predictor

# (Optional) Create and activate a virtual env
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎯 Usage

### 🤖 Training & Saving Model

Train a default Random Forest pipeline and save the model:

```bash
python train_and_save.py
```

Outputs `best_model.pkl` in the root directory.

### 🔬 Running the Evaluation Pipeline

Compare Random Forest vs. XGBoost with hyperparameter tuning:

```bash
python run_pipeline.py
```

### 🌐 Starting the API Server

Launch the FastAPI server for production-ready predictions:

```bash
uvicorn app.main:app --reload
```

- API docs: http://127.0.0.1:8000/docs

### 🖼️ Launching the Streamlit UI

Start the interactive dashboard:

```bash
streamlit run streamlit_app.py
```

The UI will attempt to call the FastAPI endpoint; if unavailable, it falls back to the saved model.

---

## 📂 Project Structure

```
.
├── app
│   └── main.py               # FastAPI application
├── model_utils.py            # Preprocessing, pipelines, evaluation utilities
├── train_and_save.py         # Train simple pipeline & save model
├── run_pipeline.py           # RF vs XGB hyperparameter tuning
├── streamlit_app.py          # Streamlit user interface
├── Random_Forset.py          # Kaggle notebook for exploratory analysis
├── spotify_history.csv       # Streaming history dataset
├── spotify_data_dictionary.csv # Dataset dictionary
├── best_model.pkl            # Saved trained model
├── requirements.txt          # Project dependencies
└── image.png                 # Logo for UI
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check [issues page](#) or submit a pull request.

---

## 🙏 Acknowledgements

- Dataset by [arshmankhalid](https://www.kaggle.com/arshmankhalid)  
- Built with [Scikit-Learn](https://scikit-learn.org/), [XGBoost](https://xgboost.ai/), [FastAPI](https://fastapi.tiangolo.com/), and [Streamlit](https://streamlit.io/)