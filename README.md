# Part 2 — CNN Computer Vision: Manufacturing Defect Detection

## Problem Statement
Multi-class image classification to automatically detect surface
defects on manufactured products. The model classifies each product
image into one of four categories: dent, stain, scratch, or normal.

## Dataset
- **Source:** Provided synthetic dataset
- **Images:** 480 total | 120 per class | 4 classes
- **Image size:** 96×96 pixels, RGB
- **Classes:** dent, stain, scratch, normal
- **Dataset Link:** https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs

## Approach
- Loaded images using Keras load_img and normalised pixels to [0,1]
- Applied stratified 80/20 train/test split
- Built a 3-block CNN with Conv2D, MaxPooling, Dropout, and Dense layers
- Trained for 30 epochs using Adam optimizer and categorical cross-entropy

## Model Architecture
Input(96,96,3) → Conv2D(32,ReLU) → MaxPool → Conv2D(64,ReLU) →
MaxPool → Conv2D(128,ReLU) → MaxPool → Flatten →
Dense(128,ReLU) → Dropout(0.3) → Dense(4,Softmax)

## Results
See results/ folder for accuracy/loss curves and confusion matrix.
See sample_predictions/ for per-class prediction examples.

## Repository Structure
part-2-cnn-computer-vision/
├── README.md
├── notebook.ipynb
├── requirements.txt
├── sample_predictions/
│   └── prediction_outputs.png
└── results/
    ├── accuracy_loss_curves.png
    └── confusion_matrix.png