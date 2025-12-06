"""
Deployment script for Hugging Face Spaces
"""
import os
import subprocess
import json
from huggingface_hub import HfApi, create_repo

def deploy_to_huggingface():
    """Deploy Streamlit app to Hugging Face Spaces"""

    # Create space repository
    space_repo_id = "your-username/tourism-predictor-app"

    api = HfApi()

    try:
        create_repo(
            space_repo_id,
            repo_type="space",
            space_sdk="streamlit",
            exist_ok=True
        )
        print(f"Created Hugging Face Space: {space_repo_id}")
    except Exception as e:
        print(f"Space might already exist: {e}")

    # Create README for the space
    readme_content = """
---
title: Tourism Package Predictor
emoji: 🏖️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Wellness Tourism Package Predictor

Predict which customers are likely to purchase Wellness Tourism Packages.

## Features
- Real-time prediction based on customer data
- Visual probability gauge
- Actionable recommendations
- Feature importance visualization

## How to Use
1. Enter customer details in the sidebar
2. Click "Predict Purchase Probability"
3. View results and recommendations

## Model Information
- Algorithm: Random Forest Classifier
- Accuracy: ~85%
- Last updated: [Current Date]
    """

    with open("README.md", "w") as f:
        f.write(readme_content)

    print("✅ Deployment files prepared!")
    print(f"🔗 Your app will be available at: https://huggingface.co/spaces/{space_repo_id}")
    print("\nTo deploy manually:")
    print("1. Go to https://huggingface.co/spaces")
    print("2. Create new Space with Streamlit SDK")
    print("3. Upload all files from deployment folder")

if __name__ == "__main__":
    deploy_to_huggingface()
