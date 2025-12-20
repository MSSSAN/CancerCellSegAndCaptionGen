---
title: Medical Image Analysis
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
license: mit
---

# Medical Image Analysis System

AI-powered breast cancer pathology image analysis using:
- **HoVerNet** - Cell segmentation
- **U-Net** - Tissue classification  
- **EfficientNetV2-S + Transformer** - Caption generation

## Features
- 🔬 Automated cell segmentation
- 🧬 Tissue type classification (Stroma, Immune, Normal, Tumor)
- 📝 AI-generated medical captions
- 📊 Detailed statistical analysis

## Usage
1. Upload a histopathology image (PNG/JPG)
2. Click "Run Complete Analysis"
3. View segmentation results and AI-generated caption
4. Download results

## Model Information
- **Input:** 512×512 pathology images
- **Channels:** 4 (RGB + Segmentation)
- **GPU:** Requires T4 or better
