"""
Run Spotify skip-prediction pipeline comparing RandomForest and XGBoost.
"""
import model_utils as mu

def main():
    # Load and preprocess data
    df = mu.load_data('spotify_history.csv', 'spotify_data_dictionary.csv')
    df = mu.preprocess_data(df)

    # Split features and target
    X, y = mu.get_features_target(df)
    X_train, X_test, y_train, y_test = mu.split_data(X, y)

    # Random Forest pipeline
    print("=== Random Forest ===")
    grid_rf = mu.build_pipeline(classifier='rf')
    model_rf, params_rf = mu.train_and_tune(grid_rf, X_train, y_train)
    print("Best RF params:", params_rf)
    mu.evaluate_model(model_rf, X_test, y_test)
    # Skipping SHAP explanation to speed up execution. To re-enable, uncomment the next line.
    # mu.explain_model(model_rf, X_train)

    # XGBoost pipeline
    try:
        print("\n=== XGBoost ===")
        grid_xgb = mu.build_pipeline(classifier='xgb')
        model_xgb, params_xgb = mu.train_and_tune(grid_xgb, X_train, y_train)
        print("Best XGB params:", params_xgb)
        mu.evaluate_model(model_xgb, X_test, y_test)
        # Skipping SHAP explanation to speed up execution. To re-enable, uncomment the next line.
        # mu.explain_model(model_xgb, X_train)
    except ImportError as e:
        print(f"Skipping XGBoost pipeline: {e}")

if __name__ == "__main__":
    main()