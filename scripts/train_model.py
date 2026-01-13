#!/usr/bin/env python3
"""
Proper output normalization for price predictions
Key changes:
1. Removed sigmoid activation (was constraining outputs incorrectly)
2. Added separate output normalization for target prices
3. Saved output denormalization parameters for inference
4. Filters outliers (> 6000 Rs) for cleaner ranges
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
    'lookback_days': 30,
    'forecast_days': 7,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.15,
    'test_split': 0.15,
    'outlier_threshold': 6000  # Filter prices above this
}

# ============================================================================
# STEP 1: DATA LOADING & CLEANING
# ============================================================================
def load_and_clean_data(csv_path):
    """Load CSV and remove problematic rows"""
    print("📂 Loading data...")
    df = pd.read_csv(csv_path)
    
    # Convert date format
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    
    # Check for national columns
    has_national = 'national_highest_price_rs_kg' in df.columns
    if has_national:
        print("   ✓ National benchmark columns detected")
    
    # Filter for target grades
    df = df[df['grade'].isin(CONFIG['grades_to_predict'])].copy()
    
    # Remove "National" and "Average Price" districts
    df = df[df['district'].str.lower() != 'national'].copy()
    df = df[df['district'] != 'Average Price'].copy()
    
    # 🆕 FILTER OUTLIERS before training
    print(f"   Filtering outliers (> {CONFIG['outlier_threshold']} Rs)...")
    original_count = len(df)
    
    avg_outliers = len(df[df['average_price_rs_kg'] > CONFIG['outlier_threshold']])
    high_outliers = len(df[df['highest_price_rs_kg'] > CONFIG['outlier_threshold']])
    
    if avg_outliers > 0 or high_outliers > 0:
        print(f"   ⚠️  Found {avg_outliers} avg price outliers, {high_outliers} high price outliers")
    
    df = df[df['average_price_rs_kg'] <= CONFIG['outlier_threshold']].copy()
    df = df[df['highest_price_rs_kg'] <= CONFIG['outlier_threshold']].copy()
    
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        print(f"   ✓ Filtered {filtered_count} outliers ({(filtered_count/original_count)*100:.2f}%)")
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"   ✓ Total records: {len(df)}")
    print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   ✓ Districts: {df['district'].nunique()}")
    print(f"   ✓ Price range: {df['average_price_rs_kg'].min():.0f} - {df['average_price_rs_kg'].max():.0f} Rs/kg")
    
    return df, has_national

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
def engineer_features(df):
    """Create time-based and price features"""
    print("\n🔧 Engineering features...")
    
    df = df.copy()
    has_national = 'national_highest_price_rs_kg' in df.columns
    
    # Time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Sort for lag features
    df = df.sort_values(['district', 'grade', 'date']).reset_index(drop=True)
    
    # Lag features
    for lag in [1, 7, 14, 30]:
        df[f'avg_price_lag_{lag}'] = df.groupby(['district', 'grade'])['average_price_rs_kg'].shift(lag)
        df[f'high_price_lag_{lag}'] = df.groupby(['district', 'grade'])['highest_price_rs_kg'].shift(lag)
        
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
        
        if has_national:
            df[f'nat_avg_roll_mean_{window}'] = df.groupby(['district', 'grade'])['national_average_price_rs_kg'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
    
    # Price momentum
    df['price_change_7d'] = df.groupby(['district', 'grade'])['average_price_rs_kg'].pct_change(7)
    df['price_change_30d'] = df.groupby(['district', 'grade'])['average_price_rs_kg'].pct_change(30)
    
    # National comparison features
    if has_national:
        df['price_vs_national'] = (df['average_price_rs_kg'] - df['national_average_price_rs_kg']) / df['national_average_price_rs_kg']
        df['price_gap_absolute'] = df['average_price_rs_kg'] - df['national_average_price_rs_kg']
    
    # Drop NaN
    df = df.dropna().reset_index(drop=True)
    
    print(f"   ✓ Created {len(df.columns)} features")
    print(f"   ✓ Remaining records: {len(df)}")
    
    return df

# ============================================================================
# STEP 3: PREPARE SEQUENCES
# ============================================================================
def create_sequences(df, lookback, forecast):
    """Create sequences with separate output normalization"""
    print(f"\n📊 Creating sequences (lookback={lookback}, forecast={forecast})...")
    
    has_national = 'national_highest_price_rs_kg' in df.columns
    
    # Encode categoricals
    district_encoder = LabelEncoder()
    grade_encoder = LabelEncoder()
    
    df['district_encoded'] = district_encoder.fit_transform(df['district'])
    df['grade_encoded'] = grade_encoder.fit_transform(df['grade'])
    
    # Select input features
    feature_cols = [
        'district_encoded', 'grade_encoded',
        'year', 'month', 'day', 'day_of_week', 'week_of_year', 'quarter',
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'average_price_rs_kg', 'highest_price_rs_kg',
    ]
    
    if has_national:
        feature_cols.extend([
            'national_average_price_rs_kg',
            'national_highest_price_rs_kg'
        ])
    
    lag_cols = [col for col in df.columns if 'lag_' in col or 'roll_' in col or 'change_' in col or 'vs_national' in col or 'gap_' in col]
    feature_cols.extend(lag_cols)
    
    print(f"   ✓ Using {len(feature_cols)} input features")
    
    # Normalize INPUT features
    input_scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = input_scaler.fit_transform(df[feature_cols])
    
    # 🆕 DEFINE OUTPUT RANGES (already filtered, so use actual min/max)
    output_avg_min = df['average_price_rs_kg'].min()
    output_avg_max = df['average_price_rs_kg'].max()
    output_high_min = df['highest_price_rs_kg'].min()
    output_high_max = df['highest_price_rs_kg'].max()
    
    print(f"\n   📊 Output price ranges (for denormalization):")
    print(f"      Average: {output_avg_min:.2f} - {output_avg_max:.2f} Rs/kg")
    print(f"      Highest: {output_high_min:.2f} - {output_high_max:.2f} Rs/kg")
    
    def normalize_output(price, min_val, max_val):
        """Normalize output to [0, 1]"""
        return (price - min_val) / (max_val - min_val)
    
    X, y = [], []
    
    # Group by district and grade
    for (district, grade), group in df_scaled.groupby(['district', 'grade']):
        group = group.sort_values('date').reset_index(drop=True)
        
        # Get original (unscaled) for output prices
        group_original = df[
            (df['district'] == district) & 
            (df['grade'] == grade)
        ].sort_values('date').reset_index(drop=True)
        
        if len(group) < lookback + forecast:
            continue
        
        for i in range(len(group) - lookback - forecast + 1):
            # Input: scaled features
            X_seq = group[feature_cols].iloc[i:i+lookback].values
            
            # Output: ORIGINAL prices (then normalize)
            y_avg = group_original['average_price_rs_kg'].iloc[i+lookback:i+lookback+forecast].values
            y_high = group_original['highest_price_rs_kg'].iloc[i+lookback:i+lookback+forecast].values
            
            # Normalize outputs to [0, 1]
            y_avg_norm = normalize_output(y_avg, output_avg_min, output_avg_max)
            y_high_norm = normalize_output(y_high, output_high_min, output_high_max)
            
            X.append(X_seq)
            y.append(np.concatenate([y_avg_norm, y_high_norm]))
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"   ✓ Created {len(X)} sequences")
    print(f"   ✓ X shape: {X.shape}")
    print(f"   ✓ y shape: {y.shape}")
    
    # Store output denormalization parameters
    output_denorm = {
        'average_price': {
            'min': float(output_avg_min),
            'max': float(output_avg_max)
        },
        'highest_price': {
            'min': float(output_high_min),
            'max': float(output_high_max)
        }
    }
    
    return X, y, input_scaler, district_encoder, grade_encoder, feature_cols, output_denorm

# ============================================================================
# STEP 4: BUILD MODEL
# ============================================================================
def build_model(input_shape, output_shape):
    """Build LSTM model - NO sigmoid!"""
    print("\n🏗️  Building LSTM model...")
    
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.3),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_shape)  # 🆕 LINEAR output
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    print(model.summary())
    print("\n   ✅ Using LINEAR output activation (no sigmoid)")
    return model

# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================
def train_model(model, X_train, y_train, X_val, y_val):
    """Train with early stopping"""
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
    """Convert to TFLite with LSTM support"""
    print("\n📱 Converting to TFLite...")
    
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        converter._experimental_lower_tensor_list_ops = False
        
        print("   ⚙️  Converter settings:")
        print("      - TFLITE_BUILTINS + SELECT_TF_OPS (for LSTM)")
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_kb = len(tflite_model) / 1024
        
        print(f"\n✅ Saved TFLite model: {output_path}")
        print(f"   Size: {size_kb:.2f} KB ({size_kb/1024:.2f} MB)")
        
        return tflite_model
        
    except Exception as e:
        print(f"\n❌ TFLite conversion failed: {e}")
        return None

# ============================================================================
# STEP 7: SAVE ARTIFACTS
# ============================================================================
def save_metadata(scaler, district_encoder, grade_encoder, feature_cols, output_denorm, config, test_loss, test_mae, has_national):
    """Save all preprocessing artifacts"""
    print("\n💾 Saving metadata...")
    
    # Save pickle files
    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(MODELS_DIR / 'district_encoder.pkl', 'wb') as f:
        pickle.dump(district_encoder, f)
    
    with open(MODELS_DIR / 'grade_encoder.pkl', 'wb') as f:
        pickle.dump(grade_encoder, f)
    
    # Create comprehensive preprocessing.json
    preprocessing_data = {
        'scaler': {
            'min_values': scaler.data_min_.tolist(),
            'max_values': scaler.data_max_.tolist(),
            'feature_range': [0.0, 1.0]
        },
        'district_encoder': {
            'classes': district_encoder.classes_.tolist(),
            'mapping': {cls: int(i) for i, cls in enumerate(district_encoder.classes_)}
        },
        'grade_encoder': {
            'classes': grade_encoder.classes_.tolist(),
            'mapping': {cls: int(i) for i, cls in enumerate(grade_encoder.classes_)}
        },
        'feature_columns': feature_cols,
        'config': config,
        'num_features': len(feature_cols),
        'lookback_days': config['lookback_days'],
        'forecast_days': config['forecast_days'],
        'output_denormalization': output_denorm,  # 🆕 CRITICAL FIX
        'has_national_features': has_national,
        'national_features': [col for col in feature_cols if 'national' in col or 'nat_' in col or 'vs_national' in col or 'gap_' in col] if has_national else [],
        'national_feature_count': len([col for col in feature_cols if 'national' in col or 'nat_' in col or 'vs_national' in col or 'gap_' in col])
    }
    
    # Save preprocessing.json
    with open(MODELS_DIR / 'preprocessing.json', 'w') as f:
        json.dump(preprocessing_data, f, indent=2)
    
    # Save metadata.json (backward compatibility)
    metadata = {
        'config': config,
        'feature_cols': feature_cols,
        'districts': district_encoder.classes_.tolist(),
        'grades': grade_encoder.classes_.tolist(),
        'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'trained_on': datetime.now().isoformat(),
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'has_national_features': has_national
    }
    
    with open(MODELS_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("   ✓ Saved preprocessing.json (includes output_denormalization)")
    print("   ✓ Saved metadata.json")
    print("   ✓ Saved encoder pickles")

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    print("=" * 70)
    print("🌿 CINNAMON PRICE PREDICTION - FIXED TRAINING")
    print("=" * 70)
    print("Key fixes:")
    print("  - Removed sigmoid activation (was causing low predictions)")
    print("  - Added proper output normalization [0,1]")
    print("  - Saved correct output_denormalization parameters")
    print("  - Filtered outliers > 6000 Rs")
    print("=" * 70)
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Load data
    df, has_national = load_and_clean_data(DATA_PATH)
    
    # Feature engineering
    df = engineer_features(df)
    
    # Create sequences
    X, y, scaler, district_enc, grade_enc, feature_cols, output_denorm = create_sequences(
        df, CONFIG['lookback_days'], CONFIG['forecast_days']
    )
    
    # Split data
    test_size = int(len(X) * CONFIG['test_split'])
    val_size = int(len(X) * CONFIG['validation_split'])
    train_size = len(X) - test_size - val_size
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"\n📊 Data splits:")
    print(f"   Train: {len(X_train)}")
    print(f"   Val:   {len(X_val)}")
    print(f"   Test:  {len(X_test)}")
    
    # Build model
    model = build_model(
        input_shape=(CONFIG['lookback_days'], len(feature_cols)),
        output_shape=CONFIG['forecast_days'] * 2
    )
    
    # Train
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate
    print("\n📈 Evaluating on test set...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Calculate real MAE
    avg_range = output_denorm['average_price']['max'] - output_denorm['average_price']['min']
    real_mae = test_mae * avg_range
    
    print(f"   Test Loss (MSE): {test_loss:.4f}")
    print(f"   Test MAE (scaled): {test_mae:.4f}")
    print(f"   Test MAE (actual): {real_mae:.2f} Rs/kg")
    print(f"   Price range: {output_denorm['average_price']['min']:.2f} - {output_denorm['average_price']['max']:.2f} Rs/kg")
    
    # Save model
    model.save(MODELS_DIR / 'cinnamon_grades_model.h5')
    print("\n✅ Saved Keras model")
    
    # Convert to TFLite
    convert_to_tflite(model, MODELS_DIR / 'cinnamon_grades_model.tflite')
    
    # Save metadata
    save_metadata(scaler, district_enc, grade_enc, feature_cols, output_denorm, CONFIG, test_loss, test_mae, has_national)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print("\n📦 Output files:")
    print("   - cinnamon_grades_model.h5")
    print("   - cinnamon_grades_model.tflite")
    print("   - preprocessing.json (✨ with output_denormalization)")
    print("   - scaler.pkl, district_encoder.pkl, grade_encoder.pkl")
    print("   - metadata.json")
    
    if has_national:
        nat_count = len([c for c in feature_cols if 'national' in c or 'nat_' in c])
        print(f"\n✨ Trained with {nat_count} national benchmark features")
    
    print("\n🚀 Deploy the new model and preprocessing.json to fix predictions!")

if __name__ == "__main__":
    main()