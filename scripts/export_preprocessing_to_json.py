#!/usr/bin/env python3
"""
Export preprocessing artifacts to JSON for Flutter/Dart consumption
CRITICAL FIX: Uses PERCENTILE-BASED ranges (matching training script)
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# ✅ MUST MATCH train_lstm_percentile.py CONFIG
CONFIG = {
    'percentile_min': 1,
    'percentile_max': 99
}

def export_to_json():
    """Export preprocessing artifacts to JSON for Flutter"""
    print("📤 Exporting preprocessing artifacts to JSON...")
    
    # Load pickle files
    print("   Loading pickle files...")
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(MODELS_DIR / 'district_encoder.pkl', 'rb') as f:
        district_encoder = pickle.load(f)
    
    with open(MODELS_DIR / 'grade_encoder.pkl', 'rb') as f:
        grade_encoder = pickle.load(f)
    
    with open(MODELS_DIR / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load actual training data
    print("   Loading training data to extract percentile-based price ranges...")
    csv_path = DATA_DIR / "cinnamon_grades.csv"
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    
    # Filter to same data used in training
    df = df[df['district'].str.lower() != 'national'].copy()
    df = df[df['district'] != 'Average Price'].copy()
    
    # Filter outliers (same as training)
    print("   Filtering outliers (> 6000 Rs)...")
    original_count = len(df)
    
    df = df[df['average_price_rs_kg'] <= 6000].copy()
    df = df[df['highest_price_rs_kg'] <= 6000].copy()
    
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        print(f"   ✓ Filtered {filtered_count} outliers ({(filtered_count/original_count)*100:.2f}%)")
    
    # ✅ USE PERCENTILE-BASED RANGES (matching training)
    print(f"\n   Using {CONFIG['percentile_min']}th-{CONFIG['percentile_max']}th percentile ranges...")
    
    avg_price_min = float(np.percentile(df['average_price_rs_kg'], CONFIG['percentile_min']))
    avg_price_max = float(np.percentile(df['average_price_rs_kg'], CONFIG['percentile_max']))
    high_price_min = float(np.percentile(df['highest_price_rs_kg'], CONFIG['percentile_min']))
    high_price_max = float(np.percentile(df['highest_price_rs_kg'], CONFIG['percentile_max']))
    
    print(f"   ✓ Average price range: {avg_price_min:.2f} - {avg_price_max:.2f} Rs/kg")
    print(f"   ✓ Highest price range: {high_price_min:.2f} - {high_price_max:.2f} Rs/kg")
    
    # Compare with actual min/max
    actual_avg_min = float(df['average_price_rs_kg'].min())
    actual_avg_max = float(df['average_price_rs_kg'].max())
    print(f"   📊 (Actual data range: {actual_avg_min:.2f} - {actual_avg_max:.2f} Rs/kg)")
    print(f"   📊 Range reduction: {((actual_avg_max - actual_avg_min) - (avg_price_max - avg_price_min)):.2f} Rs/kg")
    
    # Check for national features
    has_national = metadata.get('has_national_features', False)
    feature_cols = metadata['feature_cols']
    national_features = [col for col in feature_cols if 'national' in col.lower() or 'nat_' in col or 'vs_national' in col or 'gap_' in col]
    
    # Export scaler parameters
    print("\n   Converting scaler parameters...")
    preprocessing = {
        'scaler': {
            'min_values': scaler.data_min_.tolist(),
            'max_values': scaler.data_max_.tolist(),
            'feature_range': [float(scaler.feature_range[0]), float(scaler.feature_range[1])]
        },
        'district_encoder': {
            'classes': district_encoder.classes_.tolist(),
            'mapping': {district: int(i) for i, district in enumerate(district_encoder.classes_)}
        },
        'grade_encoder': {
            'classes': grade_encoder.classes_.tolist(),
            'mapping': {grade: int(i) for i, grade in enumerate(grade_encoder.classes_)}
        },
        'feature_columns': feature_cols,
        'config': metadata['config'],
        'num_features': len(feature_cols),
        'lookback_days': metadata['config']['lookback_days'],
        'forecast_days': metadata['config']['forecast_days'],
        
        # ✅ PERCENTILE-BASED output denormalization ranges
        'output_denormalization': {
            'average_price': {
                'min': avg_price_min,
                'max': avg_price_max,
                'method': f'percentile_{CONFIG["percentile_min"]}-{CONFIG["percentile_max"]}'
            },
            'highest_price': {
                'min': high_price_min,
                'max': high_price_max,
                'method': f'percentile_{CONFIG["percentile_min"]}-{CONFIG["percentile_max"]}'
            }
        },
        
        # National benchmark feature info
        'has_national_features': has_national,
        'national_features': national_features,
        'national_feature_count': len(national_features)
    }
    
    # Save to JSON
    output_path = MODELS_DIR / 'preprocessing.json'
    with open(output_path, 'w') as f:
        json.dump(preprocessing, f, indent=2)
    
    print(f"\n✅ Exported {output_path}")
    print(f"   Scaler: {len(scaler.data_min_)} features")
    print(f"   Districts: {len(district_encoder.classes_)}")
    print(f"   Grades: {len(grade_encoder.classes_)}")
    print(f"   Feature columns: {len(feature_cols)}")
    print(f"\n   ✨ OUTPUT DENORMALIZATION (PERCENTILE-BASED):")
    print(f"      Average: {avg_price_min:.2f} - {avg_price_max:.2f} Rs/kg")
    print(f"      Highest: {high_price_min:.2f} - {high_price_max:.2f} Rs/kg")
    print(f"      Method: {CONFIG['percentile_min']}th-{CONFIG['percentile_max']}th percentile")
    
    if has_national:
        print(f"   ✨ National features: {len(national_features)}")
    
    return preprocessing

def export_recent_data():
    """Export last 30 days of data per district/grade combination"""
    print("\n📊 Exporting recent historical data...")
    
    # Load full dataset
    data_path = DATA_DIR / "cinnamon_grades.csv"
    df = pd.read_csv(data_path)
    
    # Check for national columns
    has_national = 'national_highest_price_rs_kg' in df.columns
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    
    # Remove "National" rows
    df = df[df['district'].str.lower() != 'national'].copy()
    
    # Sort by date
    df = df.sort_values('date')
    
    # Get last 30 records per district+grade combination
    recent = df.groupby(['district', 'grade']).tail(30).reset_index(drop=True)
    
    # Save to CSV
    output_path = MODELS_DIR / 'recent_data.csv'
    recent.to_csv(output_path, index=False)
    
    print(f"✅ Exported {output_path}")
    print(f"   Total records: {len(recent)}")
    print(f"   Districts: {recent['district'].nunique()}")
    print(f"   Grades: {recent['grade'].nunique()}")
    print(f"   Date range: {recent['date'].min()} to {recent['date'].max()}")
    
    if has_national:
        print(f"   ✨ Includes national benchmark columns")
    
    return recent

def main():
    print("=" * 70)
    print("📦 EXPORTING PREPROCESSING ARTIFACTS (PERCENTILE-BASED)")
    print("=" * 70)
    
    # Export preprocessing parameters
    preprocessing = export_to_json()
    
    # Export recent data
    recent_data = export_recent_data()
    
    print("\n" + "=" * 70)
    print("✅ EXPORT COMPLETE!")
    print("=" * 70)
    print("\n📦 Files created:")
    print("   - models/preprocessing.json (PERCENTILE-BASED ranges)")
    print("   - models/recent_data.csv")
    print(f"\n🎯 Using {CONFIG['percentile_min']}th-{CONFIG['percentile_max']}th percentile ranges")
    print("   This matches the training script for consistent denormalization!")

if __name__ == "__main__":
    main()