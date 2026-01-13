#!/usr/bin/env python3
"""
Export preprocessing artifacts to JSON for Flutter/Dart consumption
Enhanced to handle national price features
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
        'uses_national_features': metadata.get('uses_national_features', False)  # NEW
    }
    
    # Save to JSON
    output_path = MODELS_DIR / 'preprocessing.json'
    with open(output_path, 'w') as f:
        json.dump(preprocessing, f, indent=2)
    
    print(f"✅ Exported {output_path}")
    print(f"   Scaler: {len(scaler.data_min_)} features")
    print(f"   Districts: {len(district_encoder.classes_)}")
    print(f"   Grades: {len(grade_encoder.classes_)}")
    print(f"   Feature columns: {len(metadata['feature_cols'])}")
    if preprocessing['uses_national_features']:
        print(f"   🌍 Model uses national price features")
    
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
    
    # Remove "Average Price" rows
    df = df[df['district'] != 'Average Price'].copy()
    
    # Check if national columns exist
    has_national = 'national_average_price_rs_kg' in df.columns
    
    # Sort by date
    df = df.sort_values('date')
    
    # Get last 30 records per district+grade combination
    recent = df.groupby(['district', 'grade']).tail(30).reset_index(drop=True)
    
    # Save to CSV (includes national columns if they exist)
    output_path = MODELS_DIR / 'recent_data.csv'
    recent.to_csv(output_path, index=False)
    
    print(f"✅ Exported {output_path}")
    print(f"   Total records: {len(recent)}")
    print(f"   Districts: {recent['district'].nunique()}")
    print(f"   Grades: {recent['grade'].nunique()}")
    print(f"   Date range: {recent['date'].min()} to {recent['date'].max()}")
    
    if has_national:
        print(f"   🌍 National price columns included")
        # Show sample of national prices
        print(f"\n   Sample national prices (latest date):")
        latest_date = recent['date'].max()
        sample = recent[recent['date'] == latest_date][['grade', 'national_average_price_rs_kg', 'national_highest_price_rs_kg']].drop_duplicates()
        for _, row in sample.iterrows():
            print(f"      {row['grade']}: Avg={row['national_average_price_rs_kg']:.2f}, High={row['national_highest_price_rs_kg']:.2f}")
    
    return recent

def export_national_summary():
    """
    NEW: Export national price summary for easy reference
    This helps Flutter app show national trends without loading full dataset
    """
    print("\n🌍 Exporting national price summary...")
    
    import pandas as pd
    
    data_path = BASE_DIR / "data" / "cinnamon_grades.csv"
    df = pd.read_csv(data_path)
    
    # Check if national columns exist
    if 'national_average_price_rs_kg' not in df.columns:
        print("   ⚠️  No national price columns found - skipping")
        return None
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    
    # Get unique date-grade combinations with national prices
    national_summary = df[['date', 'grade', 'national_average_price_rs_kg', 'national_highest_price_rs_kg']].drop_duplicates()
    
    # Sort by date
    national_summary = national_summary.sort_values('date')
    
    # Get last 90 days of national data
    cutoff_date = national_summary['date'].max() - pd.Timedelta(days=90)
    national_summary = national_summary[national_summary['date'] >= cutoff_date]
    
    # Convert date back to string for JSON compatibility
    national_summary['date'] = national_summary['date'].dt.strftime('%d.%m.%Y')
    
    # Save as JSON for easy Flutter parsing
    output_path = MODELS_DIR / 'national_prices.json'
    
    # Convert to nested structure: {grade: [{date, avg, high}, ...]}
    national_dict = {}
    for grade in national_summary['grade'].unique():
        grade_data = national_summary[national_summary['grade'] == grade]
        national_dict[grade] = [
            {
                'date': row['date'],
                'average_price_rs_kg': float(row['national_average_price_rs_kg']),
                'highest_price_rs_kg': float(row['national_highest_price_rs_kg'])
            }
            for _, row in grade_data.iterrows()
        ]
    
    with open(output_path, 'w') as f:
        json.dump(national_dict, f, indent=2)
    
    print(f"✅ Exported {output_path}")
    print(f"   Grades: {len(national_dict)}")
    print(f"   Date range: {national_summary['date'].min()} to {national_summary['date'].max()}")
    print(f"   Total records: {len(national_summary)}")
    
    return national_dict

def main():
    print("=" * 70)
    print("📦 EXPORTING PREPROCESSING ARTIFACTS FOR FLUTTER")
    print("   Enhanced with National Price Features")
    print("=" * 70)
    
    # Export preprocessing parameters
    preprocessing = export_to_json()
    
    # Export recent data (includes national columns if available)
    recent_data = export_recent_data()
    
    # Export national price summary (NEW)
    national_summary = export_national_summary()
    
    print("\n" + "=" * 70)
    print("✅ EXPORT COMPLETE!")
    print("=" * 70)
    print("\n📦 Files created:")
    print("   - models/preprocessing.json")
    print("   - models/recent_data.csv")
    if national_summary:
        print("   - models/national_prices.json  🌍 NEW")
    print("\n🚀 These files should be deployed to your public repository")
    
    if preprocessing.get('uses_national_features'):
        print("\n💡 Note: Model uses national price features")
        print("   Your Flutter app should include national prices when making predictions")

if __name__ == "__main__":
    main()