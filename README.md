# Fraud Detection Model

## Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The dataset used is highly imbalanced, and various resampling techniques were applied to balance the data. Multiple models were trained, evaluated, and optimized for better fraud detection.

## Dataset
- The dataset contains **31 features**, including transaction amount and anonymized principal components (V1 to V28).
- The target variable (**Class**) indicates whether a transaction is fraudulent (1) or normal (0).
- Original dataset distribution:
  - **Legitimate transactions:** 31,677
  - **Fraudulent transactions:** 102

## Data Preprocessing
1. **Handling Missing Values:**
   - Dropped any rows containing missing values.
2. **Feature Scaling:**
   - Applied **StandardScaler** to normalize feature values.
3. **Balancing the Data:**
   - **Under-sampling:** Reduced the majority class (legitimate transactions) to 10,000 samples.
   - **Over-sampling:** Applied **SMOTE** to balance fraudulent transactions to match legitimate transactions.

## Model Training & Evaluation
Several machine learning models were trained and optimized using GridSearchCV:

### **1. Logistic Regression**
- Accuracy on Training Data: **86.52%**
- Accuracy on Test Data: **86.27%**

### **2. Support Vector Machine (SVM)**
- Best Parameters: `{C: 10, gamma: 'scale', kernel: 'rbf'}`
- Best Cross-validation Accuracy: **99.63%**
- Test Accuracy: **99.75%**

### **3. Decision Tree Classifier**
- Best Parameters: `{criterion: 'entropy', max_depth: 10, min_samples_split: 2}`
- Best Cross-validation Accuracy: **99.02%**
- Test Accuracy: **99.02%**

### **4. Random Forest Classifier (Best Model)**
- Best Parameters: `{max_depth: None, min_samples_split: 2, n_estimators: 200}`
- Best Cross-validation Accuracy: **99.57%**
- Test Accuracy: **99.75%**

## Installation & Usage
### Prerequisites
- Python 3.x
- Jupyter Notebook / Google Colab
- Required Libraries:
  ```bash
  pip install numpy pandas scikit-learn imbalanced-learn
  ```

### Running the Model
1. Clone the repository:
   ```bash
   git clone https://github.com/NoumanYousaf14/Fraud-Detection.git
   cd Fraud-Detection
   ```
2. Open and run `model.ipynb` in Jupyter Notebook or Google Colab.

## Results & Conclusion
- **Random Forest Classifier** performed the best with a **99.75% accuracy** on test data.
- **Resampling techniques (SMOTE + Under-sampling)** significantly improved fraud detection.
- The model is effective for identifying fraudulent transactions with high precision.

## Future Improvements
- Experimenting with **deep learning models** (e.g., LSTMs, Autoencoders).
- Implementing real-time fraud detection pipelines.
- Enhancing feature engineering techniques.


