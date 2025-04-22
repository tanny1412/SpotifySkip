"""
Train a simple RandomForest pipeline on the Spotify history data and save the trained model.
"""
import pickle
import time

import model_utils as mu
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

def main():
    # Load and preprocess the full dataset
    df = mu.load_data('spotify_history.csv', 'spotify_data_dictionary.csv')
    df = mu.preprocess_data(df)
    print(f"Dataset after preprocessing: {df.shape[0]} samples, {df.shape[1]} features (including target)")
    X, y = mu.get_features_target(df)
    X_train, X_test, y_train, y_test = mu.split_data(X, y)
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Define a simple pipeline without hyperparameter tuning for fast training
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
    ])

    # Train and save the model
    print("Training pipeline on full data...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds")
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("Model trained and saved to best_model.pkl")

if __name__ == '__main__':
    main()