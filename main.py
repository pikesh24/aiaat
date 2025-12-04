# MASTER SCRIPT: House Price Prediction Comparison
# ------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("--- 1. Loading Data ---")
    # Fetch data from Scikit-Learn (Built-in)
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['PRICE'] = data.target  # Target is in units of 100k
    print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # --- 2. Splitting Data ---
    print("\n--- 2. Splitting Data (80% Train, 20% Test) ---")
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Preprocessing (Scaling) ---
    # Essential for Gradient Descent (SGD). Less critical for RF, but good practice.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data Scaled using StandardScaler (Z-score normalization).")

    # --- 4. Training Models ---
    print("\n--- 4. Training Models ---")

    # Model A: Linear Regression (Normal Equation)
    print("Training Linear Regression (Normal Equation)...")
    lr_normal = LinearRegression()
    lr_normal.fit(X_train_scaled, y_train)

    # Model B: Linear Regression (Gradient Descent)
    # We use SGDRegressor. Max_iter is the number of "steps" down the gradient.
    print("Training Linear Regression (Gradient Descent)...")
    lr_gd = SGDRegressor(max_iter=5000, tol=1e-3, eta0=0.01, random_state=42)
    lr_gd.fit(X_train_scaled, y_train)

    # Model C: Random Forest
    print("Training Random Forest (100 Trees)...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train) # Note: RF can handle unscaled data (X_train)

    # --- 5. Evaluation ---
    print("\n--- 5. Evaluation Metrics ---")
    
    def evaluate(model, X_input, y_true, name):
        preds = model.predict(X_input)
        mse = mean_squared_error(y_true, preds)
        r2 = r2_score(y_true, preds)
        print(f"[{name}] RMSE: {np.sqrt(mse):.4f} | R2 Score: {r2:.4f}")
        return preds

    # Evaluate all three
    preds_normal = evaluate(lr_normal, X_test_scaled, y_test, "LR Normal Eq")
    preds_gd = evaluate(lr_gd, X_test_scaled, y_test, "LR Gradient Desc")
    preds_rf = evaluate(rf_model, X_test, y_test, "Random Forest")

    # --- 6. Visualization ---
    print("\n--- 6. Generating Plots ---")
    plt.figure(figsize=(15, 6))

    # Plot 1: Linear Regression (Normal) vs Actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, preds_normal, alpha=0.3, color='blue', label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
    plt.xlabel("Actual Price ($100k)")
    plt.ylabel("Predicted Price ($100k)")
    plt.title(f"Linear Regression\n(R2: {r2_score(y_test, preds_normal):.2f})")
    plt.legend()

    # Plot 2: Random Forest vs Actual
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, preds_rf, alpha=0.3, color='green', label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
    plt.xlabel("Actual Price ($100k)")
    plt.ylabel("Predicted Price ($100k)")
    plt.title(f"Random Forest\n(R2: {r2_score(y_test, preds_rf):.2f})")
    plt.legend()

    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()