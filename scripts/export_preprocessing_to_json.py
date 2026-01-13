#!/usr/bin/env python3
"""
Export preprocessing artifacts to JSON for Flutter/Dart consumption
Now includes output_denormalization for correct price predictions
"""
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

def export_to_json():
    """Export preprocessing artifacts to JSON for Flutter/Dart consumption"""
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
    
    # CRITICAL: Load actual training data to get REAL price ranges
    print("   Loading training data to extract actual price ranges...")
    csv_path = DATA_DIR / "cinnamon_grades.csv"
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    
    # Filter to same data used in training (exclude "National" and "Average Price")
    df = df[df['district'].str.lower() != 'national'].copy()
    df = df[df['district'] != 'Average Price'].copy()
    
    # Get ACTUAL price ranges from the data (what model was trained on)
    avg_price_min = float(df['average_price_rs_kg'].min())
    avg_price_max = float(df['average_price_rs_kg'].max())
    high_price_min = float(df['highest_price_rs_kg'].min())
    high_price_max = float(df['highest_price_rs_kg'].max())
    
    print(f"   ✓ Actual average price range: {avg_price_min:.2f} - {avg_price_max:.2f}")
    print(f"   ✓ Actual highest price range: {high_price_min:.2f} - {high_price_max:.2f}")
    
    # Check for national features
    has_national = metadata.get('has_national_features', False)
    feature_cols = metadata['feature_cols']
    national_features = [col for col in feature_cols if 'national' in col.lower() or 'nat_' in col or 'vs_national' in col or 'gap_' in col]
    
    # Export scaler parameters
    print("   Converting scaler parameters...")
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
        
        # CRITICAL: Separate output denormalization ranges (from actual data)
        'output_denormalization': {
            'average_price': {
                'min': avg_price_min,
                'max': avg_price_max
            },
            'highest_price': {
                'min': high_price_min,
                'max': high_price_max
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
    
    print(f"✅ Exported {output_path}")
    print(f"   Scaler: {len(scaler.data_min_)} features")
    print(f"   Districts: {len(district_encoder.classes_)}")
    print(f"   Grades: {len(grade_encoder.classes_)}")
    print(f"   Feature columns: {len(feature_cols)}")
    print(f"")
    print(f"   ✨ OUTPUT DENORMALIZATION:")
    print(f"      Average: {avg_price_min:.2f} - {avg_price_max:.2f} Rs/kg")
    print(f"      Highest: {high_price_min:.2f} - {high_price_max:.2f} Rs/kg")
    
    if has_national:
        print(f"   ✨ National features: {len(national_features)}")
    
    return preprocessing

if __name__ == "__main__":
    export_to_json()