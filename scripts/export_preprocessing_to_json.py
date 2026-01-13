#!/usr/bin/env python3
"""
Export preprocessing artifacts to JSON for Flutter/Dart consumption
UPDATED: Handles National as a regular district
"""

import pickle
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

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
        'feature_columns': metadata['feature_cols'],
        'config': metadata['config'],
        'num_features': len(metadata['feature_cols']),
        'lookback_days': metadata['config']['lookback_days'],
        'forecast_days': metadata['config']['forecast_days'],
        'uses_national_features': metadata.get('uses_national_features', False)  # NEW: Flag
    }
    
    # Save to JSON
    output_path = MODELS_DIR / 'preprocessing.json'
    with open(output_path, 'w') as f:
        json.dump(preprocessing, f, indent=2)
    
    print(f"✅ Exported {output_path}")
    print(f"   Scaler: {len(scaler.data_min_)} features")
    print(f"   Districts: {len(district_encoder.classes_)}")
    
    # Check for National district
    if 'National' in district_encoder.classes_:
        print(f"   ✨ National district: INCLUDED")
    
    print(f"   Grades: {len(grade_encoder.classes_)}")
    print(f"   Feature columns: {len(metadata['feature_cols'])}")
    
    return preprocessing

def export_recent_data():
    """Export last 30 days of data per district/grade combination"""
    print("\n📊 Exporting recent historical data...")
    
    import pandas as pd
    
    # Load full dataset
    data_path = BASE_DIR / "data" / "cinnamon_grades.csv"
    df = pd.read_csv(data_path)
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    
    # Remove "Average Price" rows (if they exist)
    original_len = len(df)
    df = df[df['district'] != 'Average Price'].copy()
    removed = original_len - len(df)
    if removed > 0:
        print(f"   Removed {removed} 'Average Price' aggregate rows")
    
    # KEEP National district (it's now a regular row)
    
    # Sort by date
    df = df.sort_values('date')
    
    # Get last 30 records per district+grade combination (includes National)
    recent = df.groupby(['district', 'grade']).tail(30).reset_index(drop=True)
    
    # Save to CSV
    output_path = MODELS_DIR / 'recent_data.csv'
    recent.to_csv(output_path, index=False)
    
    print(f"✅ Exported {output_path}")
    print(f"   Total records: {len(recent)}")
    print(f"   Districts: {recent['district'].nunique()}")
    
    # Check for National records
    national_count = len(recent[recent['district'] == 'National'])
    if national_count > 0:
        print(f"   ✨ National benchmark records: {national_count}")
    
    print(f"   Grades: {recent['grade'].nunique()}")
    print(f"   Date range: {recent['date'].min()} to {recent['date'].max()}")
    
    return recent

def main():
    print("=" * 70)
    print("📦 EXPORTING PREPROCESSING ARTIFACTS FOR FLUTTER")
    print("=" * 70)
    
    # Export preprocessing parameters
    preprocessing = export_to_json()
    
    # Export recent data
    recent_data = export_recent_data()
    
    print("\n" + "=" * 70)
    print("✅ EXPORT COMPLETE!")
    print("=" * 70)
    print("\n📦 Files created:")
    print("   - models/preprocessing.json")
    print("   - models/recent_data.csv")
    print("\n🚀 These files should be deployed to your public repository")
    print("\n💡 Note: National prices are included as district='National'")

if __name__ == "__main__":
    main()