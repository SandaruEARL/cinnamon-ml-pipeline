#!/usr/bin/env python3
"""
Export preprocessing artifacts to JSON for Flutter/Dart consumption
UPDATED: Handles National as a regular district
UPDATED: Exports WEEKLY model metadata (lookback_weeks, forecast_weeks)
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
    
    # Check if this is a weekly or daily model
    is_weekly = metadata.get('data_frequency') == 'weekly'
    
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
        'uses_national_features': metadata.get('uses_national_features', False),
        'data_frequency': metadata.get('data_frequency', 'daily'),  # NEW: weekly or daily
    }
    
    # Add appropriate lookback/forecast based on model type
    if is_weekly:
        preprocessing['lookback_weeks'] = metadata['config']['lookback_weeks']
        preprocessing['forecast_weeks'] = metadata['config']['forecast_weeks']
        print("   ✨ WEEKLY MODEL DETECTED")
        print(f"      Lookback: {metadata['config']['lookback_weeks']} weeks")
        print(f"      Forecast: {metadata['config']['forecast_weeks']} weeks")
    else:
        preprocessing['lookback_days'] = metadata['config'].get('lookback_days', 30)
        preprocessing['forecast_days'] = metadata['config'].get('forecast_days', 7)
    
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
    """Export recent data based on model type (weekly or daily)"""
    print("\n📊 Exporting recent historical data...")
    
    import pandas as pd
    
    # Load metadata to check model type
    with open(MODELS_DIR / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    is_weekly = metadata.get('data_frequency') == 'weekly'
    
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
    
    if is_weekly:
        # For weekly model: resample to weekly first, then get last 12 weeks
        print("   📅 Resampling to weekly frequency for export...")
        df = df.set_index('date')
        
        weekly_data = []
        for (district, grade), group in df.groupby(['district', 'grade']):
            weekly = group.resample('W').agg({
                'average_price_rs_kg': 'last',
                'highest_price_rs_kg': 'last'
            }).reset_index()
            
            weekly['district'] = district
            weekly['grade'] = grade
            weekly['average_price_rs_kg'] = weekly['average_price_rs_kg'].ffill()
            weekly['highest_price_rs_kg'] = weekly['highest_price_rs_kg'].ffill()
            weekly = weekly.dropna()
            
            weekly_data.append(weekly)
        
        df = pd.concat(weekly_data, ignore_index=True)
        df = df.sort_values('date')
        
        # Get last 12 weeks per district+grade
        recent = df.groupby(['district', 'grade']).tail(12).reset_index(drop=True)
        print(f"   ✨ Exported last 12 WEEKS per district/grade")
    else:
        # For daily model: get last 30 records per district+grade
        recent = df.groupby(['district', 'grade']).tail(30).reset_index(drop=True)
        print(f"   Exported last 30 records per district/grade")
    
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
    print("\n💡 Notes:")
    print("   - National prices are included as district='National'")
    
    if preprocessing.get('data_frequency') == 'weekly':
        print("   - Model is WEEKLY: predicts next 4 weeks")
        print(f"   - Lookback: {preprocessing['lookback_weeks']} weeks")
        print(f"   - Forecast: {preprocessing['forecast_weeks']} weeks")
    else:
        print("   - Model is DAILY: predicts next 7 days")

if __name__ == "__main__":
    main()