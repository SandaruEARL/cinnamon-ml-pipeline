#!/usr/bin/env python3
"""
CINNAMON GRADES PRICE PREDICTION MODEL (WITH NATIONAL VALUES)
Predicts: Alba, C-5 Sp, C-5, C-4 (7 days ahead, per district)
Now includes national benchmark prices as features
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
# CONFIGURATION
# ============================================================================
CONFIG = {
    'grades_to_predict': [
        'Alba', 
        'C-4', 'C-5', 'C-5 Sp',
        'M-4', 'M-5',
        'H-1', 'H-2',
        'H-Faq', 'Heen', 'Gorosu'
    ], 
    'lookback_days': 30,      # Use past 30 days
    'forecast_days': 7,        # Predict 7 days ahead
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
    """Load CSV and remove problematic rows"""
    print("📂 Loading data...")
    df = pd.read_csv(csv_path)
    
    # Convert date format DD.MM.YYYY to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    
    # Check if national columns exist
    has_national = 'national_highest_price_rs_kg' in df.columns
    if has_national:
        print("   ✓ National benchmark columns detected")
    else:
        print("   ⚠️  No national columns found - will proceed without them")
    
    # Filter for target grades only
    df = df[df['grade'].isin(CONFIG['grades_to_predict'])].copy()
    print(f"   ✓ Filtered to grades: {CONFIG['grades_to_predict']}")
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"   ✓ Total records: {len(df)}")
    print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   ✓ Districts: {df['district'].nunique()}")
    print(f"   ✓ Unique districts: {sorted(df['district'].unique())}")
    
    return df

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
def engineer_features(df):
    """Create time-based and price features"""
    print("\n🔧 Engineering features...")
    
    df = df.copy()
    
    # Check for national columns
    has_national = 'national_highest_price_rs_kg' in df.columns
    
    # Time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    
    # Cyclical encoding for seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Sort for lag features
    df = df.sort_values(['district', 'grade', 'date']).reset_index(drop=True)
    
    # Lag features (previous prices)
    for lag in [1, 7, 14, 30]:
        df[f'avg_price_lag_{lag}'] = df.groupby(['district', 'grade'])['average_price_rs_kg'].shift(lag)
        df[f'high_price_lag_{lag}'] = df.groupby(['district', 'grade'])['highest_price_rs_kg'].shift(lag)
        
        # NEW: National price lags
        if has_national:
            df[f'nat_avg_lag_{lag}'] = df.groupby(['district', 'grade'])['national_average_price_rs_kg'].shift(lag)
            df[f'nat_high_lag_{lag}'] = df.groupby(['district', 'grade'])['national_highest_price_rs_kg'].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'avg_price_roll_mean_{window}'] = df.groupby(['district', 'grade'])['average_price_rs_kg'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'avg_price_roll_std_{window}'] = df.groupby(['district', 'grade'])['average_price_rs_kg'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
        
        # NEW: National rolling stats
        if has_national:
            df[f'nat_avg_roll_mean_{window}'] = df.groupby(['district', 'grade'])['national_average_price_rs_kg'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
    
    # Price momentum
    df['price_change_7d'] = df.groupby(['district', 'grade'])['average_price_rs_kg'].pct_change(7)
    df['price_change_30d'] = df.groupby(['district', 'grade'])['average_price_rs_kg'].pct_change(30)
    
    # NEW: Price deviation from national average
    if has_national:
        df['price_vs_national'] = (df['average_price_rs_kg'] - df['national_average_price_rs_kg']) / df['national_average_price_rs_kg']
        df['price_gap_absolute'] = df['average_price_rs_kg'] - df['national_average_price_rs_kg']
        print("   ✓ Created national benchmark features")
    
    # Drop rows with NaN from lag features
    df = df.dropna().reset_index(drop=True)
    
    print(f"   ✓ Created {len(df.columns)} features")
    print(f"   ✓ Remaining records: {len(df)}")
    
    return df

# ============================================================================
# STEP 3: PREPARE SEQUENCES FOR LSTM
# ============================================================================
def create_sequences(df, lookback, forecast):
    """Create sequences for time-series prediction"""
    print(f"\n📊 Creating sequences (lookback={lookback}, forecast={forecast})...")
    
    # Check for national columns
    has_national = 'national_highest_price_rs_kg' in df.columns
    
    # Encode categorical variables
    district_encoder = LabelEncoder()
    grade_encoder = LabelEncoder()
    
    df['district_encoded'] = district_encoder.fit_transform(df['district'])
    df['grade_encoded'] = grade_encoder.fit_transform(df['grade'])
    
    # Select features
    feature_cols = [
        'district_encoded', 'grade_encoded',
        'year', 'month', 'day', 'day_of_week', 'week_of_year', 'quarter',
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'average_price_rs_kg', 'highest_price_rs_kg',
    ]
    
    # Add national columns if they exist
    if has_national:
        feature_cols.extend([
            'national_average_price_rs_kg',
            'national_highest_price_rs_kg'
        ])
    
    # Add lag features
    lag_cols = [col for col in df.columns if 'lag_' in col or 'roll_' in col or 'change_' in col or 'vs_national' in col or 'gap_' in col]
    feature_cols.extend(lag_cols)
    
    print(f"   ✓ Using {len(feature_cols)} features")
    if has_national:
        national_features = [col for col in feature_cols if 'national' in col or 'nat_' in col or 'vs_national' in col or 'gap_' in col]
        print(f"   ✓ Including {len(national_features)} national-related features")
    
    # Normalize features
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    X, y = [], []
    
    # Group by district and grade
    for (district, grade), group in df_scaled.groupby(['district', 'grade']):
        group = group.sort_values('date').reset_index(drop=True)
        
        if len(group) < lookback + forecast:
            continue
        
        for i in range(len(group) - lookback - forecast + 1):
            # Input: past 'lookback' days
            X_seq = group[feature_cols].iloc[i:i+lookback].values
            
            # Target: next 'forecast' days (avg and high prices)
            y_avg = group['average_price_rs_kg'].iloc[i+lookback:i+lookback+forecast].values
            y_high = group['highest_price_rs_kg'].iloc[i+lookback:i+lookback+forecast].values
            
            X.append(X_seq)
            y.append(np.concatenate([y_avg, y_high]))  # Shape: (14,) for 7 days
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   ✓ Created {len(X)} sequences")
    print(f"   ✓ X shape: {X.shape}")  # (samples, lookback, features)
    print(f"   ✓ y shape: {y.shape}")  # (samples, forecast*2)
    
    return X, y, scaler, district_encoder, grade_encoder, feature_cols

# ============================================================================
# STEP 4: BUILD LSTM MODEL
# ============================================================================
def build_model(input_shape, output_shape):
    """Build LSTM model for multi-step forecasting"""
    print("\n🏗️  Building LSTM model...")
    
    model = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=input_shape),  # 128→64
    layers.Dropout(0.3),  # 0.2→0.3
    layers.LSTM(32, return_sequences=False),  # 64→32
    layers.Dropout(0.3),  # 0.2→0.3
    layers.Dense(32, activation='relu'),  # 64→32
    layers.Dense(output_shape, activation='sigmoid')  # SIGMOID activated
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
        
        return tflite_model
        
    except Exception as e:
        print(f"\n❌ TFLite conversion failed: {str(e)}")
        print("\n💡 The Keras model (.h5) was saved and can be used with TensorFlow")
        return None

# ============================================================================
# STEP 7: SAVE METADATA
# ============================================================================
def save_metadata(scaler, district_encoder, grade_encoder, feature_cols, config, test_loss, test_mae, has_national):
    """Save preprocessing artifacts and config"""
    print("\n💾 Saving metadata...")
    
    # Save encoders and scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/district_encoder.pkl', 'wb') as f:
        pickle.dump(district_encoder, f)
    
    with open('models/grade_encoder.pkl', 'wb') as f:
        pickle.dump(grade_encoder, f)
    
    # Save config and feature list
    metadata = {
        'config': config,
        'feature_cols': feature_cols,
        'districts': district_encoder.classes_.tolist(),
        'grades': grade_encoder.classes_.tolist(),
        'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'trained_on': datetime.now().isoformat(),
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'has_national_features': has_national  # NEW: Flag for national features
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("   ✓ Saved preprocessing artifacts")

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    print("=" * 70)
    print("🌿 CINNAMON GRADES PRICE PREDICTION - TRAINING PIPELINE")
    print("   (Now with National Benchmark Features)")
    print("=" * 70)
    
    # Create output directory
    import os
    os.makedirs('models', exist_ok=True)
    
    csv_path = DATA_DIR / "cinnamon_grades.csv"
    
    MODELS_DIR.mkdir(exist_ok=True)

    # Load and clean data
    df = load_and_clean_data(DATA_PATH)
    has_national = 'national_highest_price_rs_kg' in df.columns
    
    # Feature engineering
    df = engineer_features(df)
    
    # Create sequences
    X, y, scaler, district_enc, grade_enc, feature_cols = create_sequences(
        df, CONFIG['lookback_days'], CONFIG['forecast_days']
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
        input_shape=(CONFIG['lookback_days'], len(feature_cols)),
        output_shape=CONFIG['forecast_days'] * 2
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
    save_metadata(scaler, district_enc, grade_enc, feature_cols, CONFIG, test_loss, test_mae, has_national)

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
    
    if has_national:
        print("\n✨ Model trained with national benchmark features!")
        national_feature_count = len([col for col in feature_cols if 'national' in col or 'nat_' in col or 'vs_national' in col])
        print(f"   National features used: {national_feature_count}")

if __name__ == "__main__":
    main()