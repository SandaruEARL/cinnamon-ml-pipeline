#!/usr/bin/env python3
"""
Export preprocessing artifacts to JSON for Flutter/Dart consumption
Now includes national benchmark feature information
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
        
        # NEW: National benchmark feature info
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
    
    # Print national feature info
    if has_national:
        print(f"   ✨ National features: {len(national_features)}")
        print(f"      Features: {', '.join(national_features[:5])}...")
    else:
        print(f"   ℹ️  No national features (using older dataset)")
    
    return preprocessing

def export_recent_data():
    """Export last 30 days of data per district/grade combination"""
    print("\n📊 Exporting recent historical data...")
    
    import pandas as pd
    
    # Load full dataset
    data_path = BASE_DIR / "data" / "cinnamon_grades.csv"
    df = pd.read_csv(data_path)
    
    # Check for national columns
    has_national = 'national_highest_price_rs_kg' in df.columns
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    
    # Remove "National" rows (they're stored in separate columns now)
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
        # Show sample of national values
        national_cols = ['national_highest_price_rs_kg', 'national_average_price_rs_kg']
        non_null_national = recent[national_cols].notna().any(axis=1).sum()
        print(f"   Records with national data: {non_null_national}/{len(recent)}")
    else:
        print(f"   ℹ️  No national benchmark data (older dataset)")
    
    return recent

def main():
    print("=" * 70)
    print("📦 EXPORTING PREPROCESSING ARTIFACTS FOR FLUTTER")
    print("   (With National Benchmark Feature Support)")
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
    
    if preprocessing['has_national_features']:
        print("\n✨ National Features Summary:")
        print(f"   Total features: {preprocessing['num_features']}")
        print(f"   National-related: {preprocessing['national_feature_count']}")
        print(f"   Feature names: {', '.join(preprocessing['national_features'][:5])}...")
    
    print("\n🚀 These files should be deployed to your public repository")

if __name__ == "__main__":
    main()