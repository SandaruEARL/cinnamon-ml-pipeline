#!/usr/bin/env python3
"""Generate version.json for GitHub Pages deployment"""

import json
import hashlib
import os
from datetime import datetime
import pandas as pd

print("📋 Generating version info...")

# Read training metadata
with open('models/metadata.json', 'r') as f:
    metadata = json.load(f)

# Get dataset info
df = pd.read_csv('data/cinnamon_grades.csv')

# Calculate model hash
with open('models/cinnamon_grades_model.tflite', 'rb') as f:
    model_bytes = f.read()
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    model_size = len(model_bytes)

version = datetime.now().strftime('%Y%m%d_%H%M%S')

# Get GitHub info
github_repo = os.getenv('GITHUB_REPOSITORY', 'yourusername/cinnamon-ml-pipeline')
username = github_repo.split('/')[0]
repo_name = github_repo.split('/')[1]

base_url = f"https://{username}.github.io/{repo_name}"

# Parse dates
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')

version_info = {
    "version": version,
    "updated_at": datetime.now().isoformat(),
    
    # GitHub Pages URLs
    "model_url": f"{base_url}/cinnamon_grades_model.tflite",
    "version_url": f"{base_url}/version.json",
    
    # Model info
    "model_hash": model_hash,
    "model_size_bytes": model_size,
    "model_size_kb": round(model_size / 1024, 2),
    
    # Dataset info
    "records_count": len(df),
    "date_range": {
        "start": df['date'].min().strftime('%Y-%m-%d'),
        "end": df['date'].max().strftime('%Y-%m-%d')
    },
    "districts": sorted(df['district'].unique().tolist()),
    "grades": sorted(df['grade'].unique().tolist()),
    
    # Training info
    "training": {
        "test_loss": metadata.get('test_loss'),
        "test_mae": metadata.get('test_mae'),
        "trained_on": metadata.get('trained_on'),
        "lookback_days": metadata['config']['lookback_days'],
        "forecast_days": metadata['config']['forecast_days']
    },
    
    # Source
    "source_repo": f"https://github.com/{github_repo}",
    "deployment": "GitHub Pages",
    "lstm_support": "Requires SELECT_TF_OPS in TFLite"
}

with open('models/version.json', 'w') as f:
    json.dump(version_info, f, indent=2)

print(f"✓ Generated version.json")
print(f"  Version: {version}")
print(f"  Model URL: {version_info['model_url']}")
print("✅ Done!")