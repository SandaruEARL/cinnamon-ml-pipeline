#!/usr/bin/env python3
"""
CINNAMON GRADES PRICE PREDICTION MODEL - WEEKLY VERSION
Predicts: Next 4 WEEKS (not 7 days)
Data frequency: 4 reports per month (weekly)
UPDATED: Handles National as a regular district row
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "cinnamon_grades.csv"
MODELS_DIR = BASE_DIR / "models"

# ============================================================================
# CONFIGURATION - WEEKLY MODEL
# ============================================================================
CONFIG = {
    'grades_to_predict': [
        'Alba', 
        'C-4', 'C-5', 'C-5 Sp',
        'M-4', 'M-5',
        'H-1', 'H-2',
        'H-Faq', 'Heen', 'Gorosu'
    ], 
    'lookback_weeks': 12,      # Use past 12 weeks (~3 months)
    'forecast_weeks': 4,       # Predict next 4 weeks (1 month ahead)
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.15,
    'test_split': 0.15
}

# ============================================================================
# STEP 1: DATA LOADING & CLEANING
# ============================================================================
def load_and_clean_data(csv_path):
    """Load CSV and resample to weekly frequency"""
    print("📂 Loading data...")
    df = pd.read_csv(csv_path)
    
    # Convert date format DD.MM.YYYY to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    
    # CRITICAL: Remove "Average Price" aggregate rows (if they exist)
    original_len = len(df)
    df = df[df['district'] != 'Average Price'].copy()
    removed = original_len - len(df)
    if removed > 0:
        print(f"   ✓ Removed {removed} 'Average Price' aggregate rows")
    
    # KEEP "National" as a valid district
    
    # Filter for target grades only
    df = df[df['grade'].isin(CONFIG['grades_to_predict'])].copy()
    print(f"   ✓ Filtered to grades: {CONFIG['grades_to_predict']}")
    
    print(f"   ✓ Raw records before resampling: {len(df)}")
    
    # ============================================
    # CRITICAL: RESAMPLE TO WEEKLY FREQUENCY
    # ============================================
    print("   📅 Resampling to weekly frequency...")
    df = df.set_index('date')
    
    # Group by district and grade, then resample to weekly
    weekly_data = []
    for (district, grade), group in df.groupby(['district', 'grade']):
        # Resample to weekly (W = week ending on Sunday)
        weekly = group.resample('W').agg({
            'average_price_rs_kg': 'last',
            'highest_price_rs_kg': 'last'
        }).reset_index()
        
        # Add back district and grade
        weekly['district'] = district
        weekly['grade'] = grade
        
        # Forward fill missing weeks (if any)
        weekly['average_price_rs_kg'] = weekly['average_price_rs_kg'].ffill()
        weekly['highest_price_rs_kg'] = weekly['highest_price_rs_kg'].ffill()
        
        # Drop rows with NaN (if still exists)
        weekly = weekly.dropna()
        
        weekly_data.append(weekly)
    
    df = pd.concat(weekly_data, ignore_index=True)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"   ✓ Weekly records after resampling: {len(df)}")
    print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   ✓ Districts: {df['district'].nunique()}")
    print(f"   ✓ Unique districts: {sorted(df['district'].unique())}")
    
    # Check if National is present
    national_count = len(df[df['district'] == 'National'])
    if national_count > 0:
        print(f"   ✓ National benchmark records: {national_count}")
    
    return df

# ============================================================================
# STEP 2: FEATURE ENGINEERING - WEEKLY FEATURES
# ============================================================================
def engineer_features(df):
    """Create time-based and price features for WEEKLY data"""
    print("\n🔧 Engineering weekly features...")
    
    df = df.copy()
    
    # Time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    
    # Cyclical encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # Sort for lag features
    df = df.sort_values(['district', 'grade', 'date']).reset_index(drop=True)
    
    # ============================================
    # WEEKLY LAG FEATURES
    # ============================================
    # lag_1 = last week, lag_4 = 4 weeks ago (1 month), etc.
    for lag in [1, 4, 8, 12]:  # 1 week, 1 month, 2 months, 3 months
        df[f'avg_price_lag_{lag}'] = df.groupby(['district', 'grade'])['average_price_rs_kg'].shift(lag)
        df[f'high_price_lag_{lag}'] = df.groupby(['district', 'grade'])['highest_price_rs_kg'].shift(lag)
    
    # ============================================
    # WEEKLY ROLLING STATISTICS
    # ============================================
    for window in [4, 8, 12]:  # 4 weeks (1 month), 8 weeks (2 months), 12 weeks (3 months)
        df[f'avg_price_roll_mean_{window}'] = df.groupby(['district', 'grade'])['average_price_rs_kg'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'avg_price_roll_std_{window}'] = df.groupby(['district', 'grade'])['average_price_rs_kg'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # ============================================
    # WEEKLY PRICE MOMENTUM
    # ============================================
    df['price_change_4w'] = df.groupby(['district', 'grade'])['average_price_rs_kg'].pct_change(4)   # 1 month
    df['price_change_12w'] = df.groupby(['district', 'grade'])['average_price_rs_kg'].pct_change(12) # 3 months
    
    # Drop rows with NaN from lag features
    df = df.dropna().reset_index(drop=True)
    
    print(f"   ✓ Created {len(df.columns)} features")
    print(f"   ✓ Remaining records: {len(df)}")
    
    return df

# ============================================================================
# STEP 3: PREPARE SEQUENCES FOR LSTM - WEEKLY SEQUENCES
# ============================================================================
def create_sequences(df, lookback_weeks, forecast_weeks):
    """Create sequences for WEEKLY time-series prediction"""
    print(f"\n📊 Creating weekly sequences (lookback={lookback_weeks} weeks, forecast={forecast_weeks} weeks)...")
    
    # Encode categorical variables
    district_encoder = LabelEncoder()
    grade_encoder = LabelEncoder()
    
    df['district_encoded'] = district_encoder.fit_transform(df['district'])
    df['grade_encoded'] = grade_encoder.fit_transform(df['grade'])
    
    # Select features
    feature_cols = [
        'district_encoded', 'grade_encoded',
        'year', 'month', 'week_of_year', 'quarter',
        'month_sin', 'month_cos', 'week_sin', 'week_cos',
        'average_price_rs_kg', 'highest_price_rs_kg',
    ]
    
    # Add lag features
    lag_cols = [col for col in df.columns if 'lag_' in col or 'roll_' in col or 'change_' in col]
    feature_cols.extend(lag_cols)
    
    # Normalize features
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    X, y = [], []
    
    # Group by district and grade (includes National as a district)
    for (district, grade), group in df_scaled.groupby(['district', 'grade']):
        group = group.sort_values('date').reset_index(drop=True)
        
        if len(group) < lookback_weeks + forecast_weeks:
            continue
        
        for i in range(len(group) - lookback_weeks - forecast_weeks + 1):
            # Input: past 'lookback_weeks' weeks
            X_seq = group[feature_cols].iloc[i:i+lookback_weeks].values
            
            # Target: next 'forecast_weeks' weeks (avg and high prices)
            y_avg = group['average_price_rs_kg'].iloc[i+lookback_weeks:i+lookback_weeks+forecast_weeks].values
            y_high = group['highest_price_rs_kg'].iloc[i+lookback_weeks:i+lookback_weeks+forecast_weeks].values
            
            X.append(X_seq)
            y.append(np.concatenate([y_avg, y_high]))  # Shape: (8,) for 4 weeks
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   ✓ Created {len(X)} sequences")
    print(f"   ✓ X shape: {X.shape}")  # (samples, lookback_weeks, features)
    print(f"   ✓ y shape: {y.shape}")  # (samples, forecast_weeks*2)
    print(f"   ✓ Districts in model: {sorted(district_encoder.classes_)}")
    
    return X, y, scaler, district_encoder, grade_encoder, feature_cols

# ============================================================================
# STEP 4: BUILD LSTM MODEL
# ============================================================================
def build_model(input_shape, output_shape):
    """Build LSTM model for multi-week forecasting"""
    print("\n🏗️  Building LSTM model for WEEKLY forecasting...")
    
    model = keras.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_shape)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    print(model.summary())
    return model

# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================
def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with early stopping"""
    print("\n🚀 Training model...")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# ============================================================================
# STEP 6: CONVERT TO TFLITE
# ============================================================================
def convert_to_tflite(model, output_path):
    """Convert Keras model to TFLite format with LSTM support"""
    print("\n📱 Converting to TFLite...")
    
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        converter._experimental_lower_tensor_list_ops = False
        
        print("   ⚙️  Converter settings:")
        print("      - TFLITE_BUILTINS: enabled")
        print("      - SELECT_TF_OPS: enabled (for LSTM)")
        print("      - Tensor list lowering: disabled")
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_kb = len(tflite_model) / 1024
        size_mb = size_kb / 1024
        
        print(f"\n✅ Saved TFLite model: {output_path}")
        print(f"   Size: {size_kb:.2f} KB ({size_mb:.2f} MB)")
        
        if size_mb > 5:
            print(f"\n   ℹ️  Note: Model is larger because it includes TF ops for LSTM")
        
        return tflite_model
        
    except Exception as e:
        print(f"\n❌ TFLite conversion failed: {str(e)}")
        print("\n💡 The Keras model (.h5) was saved and can be used with TensorFlow")
        return None

# ============================================================================
# STEP 7: SAVE METADATA
# ============================================================================
def save_metadata(scaler, district_encoder, grade_encoder, feature_cols, config, test_loss, test_mae):
    """Save preprocessing artifacts and config"""
    print("\n💾 Saving metadata...")
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/district_encoder.pkl', 'wb') as f:
        pickle.dump(district_encoder, f)
    
    with open('models/grade_encoder.pkl', 'wb') as f:
        pickle.dump(grade_encoder, f)
    
    metadata = {
        'config': config,
        'feature_cols': feature_cols,
        'districts': district_encoder.classes_.tolist(),
        'grades': grade_encoder.classes_.tolist(),
        'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'trained_on': datetime.now().isoformat(),
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'uses_national_features': 'National' in district_encoder.classes_,
        'data_frequency': 'weekly',  # IMPORTANT: Mark as weekly model
        'lookback_weeks': config['lookback_weeks'],
        'forecast_weeks': config['forecast_weeks']
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("   ✓ Saved preprocessing artifacts")

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    print("=" * 70)
    print("🌿 CINNAMON GRADES PRICE PREDICTION - WEEKLY MODEL")
    print("   Predicts: NEXT 4 WEEKS (not 7 days)")
    print("   Data frequency: 4 reports per month")
    print("=" * 70)
    
    import os
    os.makedirs('models', exist_ok=True)
    
    MODELS_DIR.mkdir(exist_ok=True)

    # Load and clean data (resamples to weekly)
    df = load_and_clean_data(DATA_PATH)
    
    # Feature engineering (weekly features)
    df = engineer_features(df)
    
    # Create sequences
    X, y, scaler, district_enc, grade_enc, feature_cols = create_sequences(
        df, CONFIG['lookback_weeks'], CONFIG['forecast_weeks']
    )
    
    # Train/val/test split (time-based)
    test_size = int(len(X) * CONFIG['test_split'])
    val_size = int(len(X) * CONFIG['validation_split'])
    train_size = len(X) - test_size - val_size
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"\n📊 Data splits:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Build model
    model = build_model(
        input_shape=(CONFIG['lookback_weeks'], len(feature_cols)),
        output_shape=CONFIG['forecast_weeks'] * 2  # avg + high prices for each week
    )
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print("\n📈 Evaluating on test set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

    # UNSCALE MAE to get real price error
    price_feature_idx = feature_cols.index('average_price_rs_kg')
    price_min = scaler.data_min_[price_feature_idx]
    price_max = scaler.data_max_[price_feature_idx]
    price_range = price_max - price_min
    real_mae = test_mae * price_range

    print(f"   Test Loss (MSE): {test_loss:.4f}")
    print(f"   Test MAE (scaled): {test_mae:.4f}")
    print(f"   Test MAE (actual): {real_mae:.2f} Rs/kg")
    print(f"   Price range in data: {price_min:.2f} - {price_max:.2f} Rs/kg")
    
    # Save Keras model
    model.save('models/cinnamon_grades_model.h5')
    print("\n✅ Saved Keras model: models/cinnamon_grades_model.h5")
    
    # Convert to TFLite
    convert_to_tflite(model, 'models/cinnamon_grades_model.tflite')
    
    # Save metadata
    save_metadata(scaler, district_enc, grade_enc, feature_cols, CONFIG, test_loss, test_mae)

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print("\n📦 Output files:")
    print("   - models/cinnamon_grades_model.h5 (Keras)")
    print("   - models/cinnamon_grades_model.tflite (Mobile)")
    print("   - models/scaler.pkl")
    print("   - models/district_encoder.pkl")
    print("   - models/grade_encoder.pkl")
    print("   - models/metadata.json")
    print("\n💡 Model predicts NEXT 4 WEEKS (matches your data frequency)")
    print("   Lookback: 12 weeks | Forecast: 4 weeks")

if __name__ == "__main__":
    main()