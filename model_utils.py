import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve, auc, roc_curve)
from imblearn.over_sampling import SMOTE
# SHAP import deferred to explain_model
import shap
# Feature engineering imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# XGBoost import deferred to build_pipeline to handle missing OpenMP on macOS

def load_data(history_path='spotify_history.csv', dict_path='spotify_data_dictionary.csv'):
    """
    Load Spotify streaming history CSV file.
    """
    df = pd.read_csv(history_path, encoding='UTF-8-SIG')
    return df

def preprocess_data(df):
    """
    Feature-engineer and encode the raw Spotify history dataframe.
    """
    # Convert end timestamp to datetime
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    # Compute approximate start timestamp by subtracting play duration
    df['start_ts'] = df['ts'] - pd.to_timedelta(df['ms_played'], unit='ms')

    # Time-based features (at start of playback)
    df['hour'] = df['start_ts'].dt.hour
    df['month'] = df['start_ts'].dt.month
    # Day of week (0=Monday .. 6=Sunday)
    df['weekday'] = df['start_ts'].dt.weekday

    # Encode categoricals available at start
    df['platform'] = LabelEncoder().fit_transform(df['platform'])
    df['reason_start'] = LabelEncoder().fit_transform(df['reason_start'])

    # Binary flags
    df['shuffle'] = df['shuffle'].astype(int)
    df['skipped'] = df['skipped'].astype(int)

    # Drop columns not available or relevant at start
    df = df.drop([
        'ts', 'start_ts', 'spotify_track_uri', 'track_name',
        'artist_name', 'album_name', 'ms_played', 'reason_end'
    ], axis=1)
    return df

def get_features_target(df):
    """
    Split dataframe into features X and target y.
    """
    X = df.drop('skipped', axis=1)
    y = df['skipped']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and test sets.
    """
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)

def build_pipeline(param_grid=None, classifier='rf'):
    """
    Create a pipeline: scaler -> SMOTE -> classifier, then GridSearchCV.
    classifier: 'rf' or 'xgb'.
    """
    if classifier == 'rf':
        clf = RandomForestClassifier(random_state=42)
        default_grid = {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [10, 20, None],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4]
        }
    elif classifier == 'xgb':
        try:
            from xgboost import XGBClassifier
        except Exception:
            raise ImportError(
                "XGBoost import failed. On macOS install OpenMP runtime: brew install libomp"
            )
        clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        default_grid = {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [3, 6, 10],
            'clf__learning_rate': [0.01, 0.1],
            'clf__subsample': [0.7, 1.0]
        }
    else:
        raise ValueError("Unknown classifier type")

    # Assemble preprocessing: numeric and categorical feature pipelines
    numeric_features = ['hour', 'month', 'weekday', 'shuffle']
    categorical_features = ['platform', 'reason_start']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
    ])
    # Build the full pipeline: preprocessing -> SMOTE -> classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', clf)
    ])

    if param_grid is None:
        param_grid = default_grid

    grid = GridSearchCV(pipeline, param_grid, cv=3,
                        scoring='roc_auc', verbose=2, n_jobs=-1)
    return grid

def train_and_tune(grid, X_train, y_train):
    """
    Fit GridSearchCV and return best estimator and params.
    """
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

def evaluate_model(model, X_test, y_test, plot_curves=True):
    """
    Print accuracy, classification report, ROC AUC, PR AUC.
    Optionally plot curves.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    print(f"ROC AUC: {roc_auc:.2f}")
    print(f"PR AUC: {pr_auc:.2f}")

    if plot_curves:
        plot_roc_pr_curves(y_test, y_prob)
    return {'accuracy': acc, 'roc_auc': roc_auc, 'pr_auc': pr_auc}

def plot_roc_pr_curves(y_test, y_prob):
    """
    Plot ROC and Precision-Recall curves side by side.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

def explain_model(model, X):
    """
    Compute and plot SHAP feature importances.
    """
    try:
        # If model is a pipeline, extract classifier and optional scaler
        if hasattr(model, 'named_steps'):
            scaler = model.named_steps.get('scaler', None)
            clf = model.named_steps.get('clf', None)
            if clf is None:
                raise ValueError("Cannot find classifier step for SHAP explanation")
            # transform features if scaler is present
            X_trans = scaler.transform(X) if scaler is not None else X
            explainer = shap.Explainer(clf, X_trans)
            shap_values = explainer(X_trans)
            shap.summary_plot(shap_values, X, plot_type="bar")
        else:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            shap.summary_plot(shap_values, X, plot_type="bar")
    except Exception as e:
        print(f"SHAP explanation skipped: {e}")