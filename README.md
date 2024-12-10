# PhytoVision

**PhytoVision** is a plant leaf identification system that classifies 32 plant species using advanced image processing techniques. By analyzing shape, color, and texture features, it achieves over 98% accuracy using an SVM classifier trained on the Flavia dataset.

---

## Features
- **High Accuracy**: Achieves a classification accuracy of over 98% using an SVM classifier with an RBF kernel.
- **Advanced Feature Extraction**: Utilizes shape, color, and texture-based features for effective classification.
- **Scalability**: Modular code to add new features, preprocessors, or classifiers.
- **Preprocessing Pipeline**: Background subtraction, noise smoothing, adaptive thresholding, and morphological operations.
- **Visualization**: Confusion matrix, ROC curve, and cluster visualizations for better interpretability.

---

## System Requirements
- **Python Version**: 3.8+

### Required Libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scikit-image`
- `mahotas`
- `Pillow`
- `joblib`

### Install dependencies using:
```bash
pip install -r Requirements.txt
``` 
## Directory Structure 
```
PhytoVision/
│
├── assets/                   # Contains sample test images
│   └── test.jpg
│
├── data/                     # Contains the Flavia dataset and extracted features CSV
│   └── Flavia_features.csv
│
Flavia leaves dataset # consists of 1000 samples
├── docs/                     # Documentation and findings
│   └── model_findings.txt    # Accuracy, classification report, and confusion matrix
│
├── models/                   # Saved model and scaler files
│   ├── scaler.pkl
│   └── svm_model.pkl
│
├── scripts/                  # Jupyter notebooks for various components
│   ├── background_remover.ipynb     # Removes background
│   ├── classification.ipynb  # Classification pipeline
│   ├── preprocessing.ipynb   # Feature extraction and preprocessing
│   ├── single_preprocessing.ipynb # Preprocessing for a single image
│
├── .gitignore                # Git ignore file
├──Requirements.txt 
├── LICENSE                   # Project license
├── Main.ipynb                # Main Jupyter notebook for training, evaluation, and visualization
├── README.md                 # Project documentation

```
---

## How to Use

### 1. Preprocessing and Feature Extraction
Run `Main.ipynb` to perform preprocessing and extract features:
- Converts images to grayscale.
- Smooths images using Gaussian filtering.
- Performs adaptive thresholding.
- Extracts shape, color, and texture-based features.
- Saves the features to `data/Flavia_features.csv`.

---

### 2. Model Training
Within `main.ipynb`, train the SVM model:
- Loads the features.
- Splits the data into training and testing sets.
- Scales the features using `StandardScaler`.
- Trains the SVM classifier with an RBF kernel.
- Saves the trained model to `models/svm_model.pkl`.

---

### 3. Model Evaluation
Evaluate the model's performance:
- Displays accuracy, classification report, and confusion matrix.
- Visualizes the confusion matrix and ROC curve.
- Saves findings to `model_findings.txt`.

---

## Key Metrics
- **Model Accuracy**: ~98.08%
- **Classification Report**:
  - Precision: 97% (macro average)
  - Recall: 96% (macro average)
  - F1-Score: 96% (macro average)

---

## Visualizations
- **Confusion Matrix**: Visual representation of model performance.
- **ROC Curve**: Displays the model's tradeoff between true positive rate and false positive rate.

---

## Feature Importance Analysis
- **Principal Component Analysis (PCA)**:
- Explains variance ratios for top components.
- **Permutation Feature Importance**:
- Analyzes feature impact on the model's accuracy.

---

## Future Enhancements
- Add support for real-time classification using a webcam or mobile application.
- Expand the dataset to include more plant species.
- Implement deep learning-based classification (e.g., CNNs).
- Improve feature extraction with advanced methods like Fourier transforms or wavelets.

---

## Authors
**Deepika P.**  
Shiv Nadar University, Chennai  
Specialization: Cybersecurity  

For any queries, feel free to reach out at [your-email@example.com].

---

## Acknowledgments
- **Flavia Dataset**: Used for training and evaluation.
- **scikit-learn** and **scikit-image**: Libraries for feature extraction and classification.

---

## License
This project is licensed under the MIT License.  
Feel free to use, modify, and distribute this project as per the license terms.
