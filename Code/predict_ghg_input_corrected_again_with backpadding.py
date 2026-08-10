import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import os
import tkinter as tk
from tkinter import filedialog

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACT_DIR = os.path.join(BASE_DIR, "Artifact")
MODEL_DIR = os.path.join(BASE_DIR, "Emssion models")
PATHS = {
    "model": os.path.join(MODEL_DIR, "GRU_Best.keras"),
    "scaler": os.path.join(ARTIFACT_DIR, "minmax_scaler.joblib"),
    "lambdas": os.path.join(ARTIFACT_DIR, "yeojohnson_lambdas.json"), 
    "vocab": os.path.join(ARTIFACT_DIR, "vocabularies.json"),
    "config": os.path.join(ARTIFACT_DIR, "feature_config.json")
}

# --- 2. HELPER: INVERSE YEO-JOHNSON ---
def inverse_yeojohnson(value, lmbda):
    """Reverses the Yeo-Johnson transformation."""
    value = float(value)
    lmbda = float(lmbda)
    
    if lmbda != 0:
        base = value * lmbda + 1
        # Clip negative base to avoid NaNs, though MinMaxScaler usually prevents this
        base = max(base, 0)
        return (base)**(1/lmbda) - 1
    else:
        return np.exp(value) - 1

# --- 3. LOADING ARTIFACTS ---
def load_artifacts():
    try:
        model = tf.keras.models.load_model(PATHS["model"])
        scaler = joblib.load(PATHS["scaler"])
        
        def load_json(path):
            with open(path, 'r') as f: return json.load(f)

        lambdas = load_json(PATHS["lambdas"])
        vocab = load_json(PATHS["vocab"])
        config = load_json(PATHS["config"])
        
        return model, scaler, lambdas, vocab, config
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None

# --- 4. PREPROCESSING ---
def preprocess_input(df, scaler, lambdas, vocab, config):
    data = df.copy()
    
    # ---------------------------------------------------------
    # A. Map Categoricals (CORRECTED)
    # ---------------------------------------------------------
    cat_map = {
        "State": "State_Id",
        "Industry Type (sectors)": "Industry Type (sectors)_Id",
        "Unit Type": "Unit Type_Id",
        "Primary NAICS Code": "Primary NAICS Code_Id"
    }
    
    for csv_col, config_col in cat_map.items():
        if csv_col in data.columns:
            vocab_list = vocab.get(csv_col, [])
            
            # FIX: Do NOT add +1. 
            # The loaded vocab list already contains the placeholders at index 0 and 1.
            mapping = {str(val): i for i, val in enumerate(vocab_list)}
            
            # Map the column. Any value not in the list (unknown) gets mapped to index 1 (standard [UNK] index)
            # We use .get(x, 1) logic via map, or fillna(1)
            data[config_col] = data[csv_col].astype(str).map(mapping).fillna(1).astype(int)

    # ---------------------------------------------------------
    # B. Apply Yeo-Johnson
    # ---------------------------------------------------------
    # Using the exact same logic as training
    co2_col = "Total_CO2e"
    co2_lmbda = lambdas.get(co2_col, 0)
    
    if co2_lmbda != 0:
        data[f"{co2_col}_YeoJohnson"] = ((data[co2_col].clip(lower=0) + 1)**co2_lmbda - 1) / co2_lmbda
    else:
        data[f"{co2_col}_YeoJohnson"] = np.log(data[co2_col].clip(lower=0) + 1)
    
    cap_col = "Unit Maximum Rated Heat Input Capacity (mmBTU/hr)"
    cap_lmbda = lambdas.get(cap_col, 0)
    
    if cap_lmbda != 0:
        data[f"{cap_col}_YeoJohnson"] = ((data[cap_col].clip(lower=0) + 1)**cap_lmbda - 1) / cap_lmbda
    else:
        data[f"{cap_col}_YeoJohnson"] = np.log(data[cap_col].clip(lower=0) + 1)

    # ---------------------------------------------------------
    # C. Apply Scaling
    # ---------------------------------------------------------
    # The scaler expects columns in a specific order: [Capacity, CO2]
    # We create a temp dataframe to ensure this order
    cols_to_scale = [f'{cap_col}_YeoJohnson', f'{co2_col}_YeoJohnson']
    
    # Handle missing columns if they don't exist
    for c in cols_to_scale:
        if c not in data.columns: data[c] = 0.0

    scaled_matrix = scaler.transform(data[cols_to_scale])
    
    data['Capacity_Scaled'] = scaled_matrix[:, 0]
    data['Total_CO2e_Scaled'] = scaled_matrix[:, 1]
    
    # ---------------------------------------------------------
    # D. Final Features
    # ---------------------------------------------------------
    data['Is_Zero_Reported'] = (data['Total_CO2e'] == 0).astype(int)
    
    # Select columns exactly as the model expects
    dyn_cols = ['Total_CO2e_Scaled', 'Capacity_Scaled', 'Is_Zero_Reported']
    stat_cols = config["static_features"]
    
    # Cast to float32 for TensorFlow compatibility
    return data[dyn_cols].values.astype(np.float32), data[stat_cols].values.astype(np.float32)

# --- 5. MAIN PREDICTION ---
def run_prediction():
    artifacts = load_artifacts()
    if not artifacts: return
    model, scaler, lambdas, vocab, config = artifacts
    
    print("Please select your input CSV snippet...")
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    root.destroy()
    if not file_path: return

    raw_df = pd.read_csv(file_path)
    lookback = 6
    
    # =========================================================
    #  HANDLE DATA LENGTH (PAD OR TRUNCATE)
    # =========================================================
    n_rows = len(raw_df)
    
    if n_rows >= lookback:
        # CASE A: More than 6 years -> Take the last 6
        input_df = raw_df.tail(lookback).copy()
        
    elif n_rows < lookback:
        # CASE B: Less than 6 years -> Pad with Backfilling
        print(f"Warning: Only {n_rows} years found. Padding inputs...")
        
        missing_rows = lookback - n_rows
        
        # Take the oldest available row (the first row of the dataframe)
        first_row = raw_df.iloc[[0]].copy()
        
        # Replicate it 'missing_rows' times
        padding_df = pd.concat([first_row] * missing_rows, ignore_index=True)
        
        # Adjust 'Reporting Year' for the padded rows to keep year logic valid (optional but good for debugging)
        # We subtract years going backwards
        first_year = first_row['Reporting Year'].values[0]
        padding_df['Reporting Year'] = [first_year - (i + 1) for i in reversed(range(missing_rows))]

        # Combine: [Padding] + [Actual Data]
        input_df = pd.concat([padding_df, raw_df], ignore_index=True)

    # =========================================================
    # PREPROCESSING & PREDICTION
    # =========================================================
    # Preprocess
    dyn_data, stat_data = preprocess_input(input_df, scaler, lambdas, vocab, config)
    
    # Reshape Inputs
    # Dynamic: (1, 6, 3)
    dynamic_in = dyn_data.reshape(1, lookback, 3)
    
    # Static: (1, 4) - Using the last row's static features (Actual current data)
    static_in = stat_data[-1].reshape(1, -1)
    
    # Predict
    pred_scaled = model.predict([dynamic_in, static_in], verbose=0)[0][0]
    
    # ---------------------------------------------------------
    # INVERSE TRANSFORM
    # ---------------------------------------------------------
    dummy = np.zeros((1, 2))
    last_capacity_scaled = dyn_data[-1, 1] 
    
    dummy[0, 0] = last_capacity_scaled
    dummy[0, 1] = pred_scaled 
    
    inv_scaled_matrix = scaler.inverse_transform(dummy)
    inv_scaled_co2 = inv_scaled_matrix[0, 1]
    
    # Inverse Yeo-Johnson
    final_val = inverse_yeojohnson(inv_scaled_co2, lambdas.get("Total_CO2e", 0))
    
    print("\n" + "="*50)
    print(f"PREDICTION RESULT")
    print("="*50)
    print(f"Input Range:       {input_df['Reporting Year'].min()} - {input_df['Reporting Year'].max()}")
    print(f"Original Rows:     {n_rows}")
    print(f"Predicting Year:   {int(input_df['Reporting Year'].max()) + 1}")
    print("-" * 50)
    print(f"Projected Emission: {final_val:,.2f} MT CO2e")
    print("="*50)

if __name__ == "__main__":
    run_prediction()