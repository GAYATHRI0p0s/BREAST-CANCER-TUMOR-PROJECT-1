# BREAST-CANCER-TUMOR-PROJECT-1
# Breast Cancer Classification Project

**Student Name:** GAYATHRI PS, APARNA.V

**Course:** Msc computer science with specilaized in Data Analytics

## 1. Problem Statement

The goal is to predict whether a breast tumor is Benign or Malignant using clinical measurements of cell nuclei. By utilizing machine learning algorithms, we aim to provide a data-driven diagnostic tool that assists medical professionals in identifying breast cancer at an early stage, potentially improving patient survival rates.

## 2. Dataset Description

* **Source:** Wisconsin Breast Cancer Dataset (Kaggle).

* **Size:** 569 rows and 32 columns.

* **Target:** diagnosis (M = Malignant, B = Benign).
  
## 3. Methodology Overview

I followed the 10-stage lifecycle:

* **Cleaning:** Handled unnecessary columns (dropped id and Unnamed: 32) and encoded the target variable (M=1, B=0).
* **EDA:** Visualized feature distributions and correlation heatmaps to identify the strongest predictors of malignancy.

* **Models:** Developed and compared three primary classifiers: Logistic Regression, Support Vector Machine (SVM), and Random Forest.

* **Best Model:** **RANDOM FOREST** with ~97% accuracy.
* **Preprocessing:** Applied StandardScaler to normalize clinical features for better model convergence.

## 4. Exploratory Data Analysis
### Class Distribution
* Slightly more benign cases than malignant
* Dataset is fairly balanced
* Suitable for classification
  
![Class Distribution](EDA_distribution.png)

### Correlation Heatmap
* Strong correlation between radius, perimeter, area
* Key features influence tumor classification
* Helps understand feature relationships
  
![Correlation Heatmap](EDA_correlation.png)

## 5. Results & Comparison

| Model | Accuracy | Recall |

| :--- | :--- | :--- |

| **Random forest** | **97.37%** | **95.65%** |

| Logistic Regression | 95.62% | 91.38% |

| SVM| 96.49% | 93.48%|

## 6. Model Interpretation

The confusion matrix below shows the performance of the best-performing model (Random Forest) on the test dataset.

![Confusion Matrix](Confusion_Matrix.png)

## 7. Deployment Screenshots

### Benign Prediction
![Benign Prediction](Benign.png)

### Malignant Prediction
![Malignant Prediction](Malignant.png)

## 8. Live Deployment
[View Live Streamlit App Here](https://breast-canser-tumor-project-1-3tpqeaxb3vbvnwrdpr44hq.streamlit.app/)
