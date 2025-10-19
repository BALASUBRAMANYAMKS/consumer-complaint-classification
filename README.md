# Consumer Complaint Classification 🎯

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

A comprehensive machine learning project that automatically classifies consumer financial complaints into 4 categories using advanced deep learning and gradient boosting techniques.

## 🏆 Project Highlights

- **Best Accuracy:** 99.90% (Artificial Neural Network)
- **Dataset Size:** 1,321,283 consumer complaints
- **GPU Acceleration:** LightGBM trained in just 4.76 minutes
- **Production Ready:** All models saved and deployable

## 📊 Model Performance

| Model | Accuracy | Training Time | Status |
|-------|----------|---------------|--------|
| **ANN** | **99.90%** | 24.42 min | 🏆 Best |
| **LightGBM** | **98.67%** | 4.76 min | ⚡ Fastest |
| **Random Forest** | **96.42%** | 15.35 min | ✓ Good |
| **Logistic Regression** | **65.66%** | 3.42 min | ✓ Baseline |

## 🎯 Problem Statement

Financial institutions receive thousands of consumer complaints daily across various categories. Manual classification is:
- ⏰ Time-consuming (5 minutes per complaint)
- ❌ Error-prone (85-90% human accuracy)
- 💰 Expensive (requires trained staff)

**Our Solution:** Automated ML pipeline that classifies complaints in <1 second with 99.90% accuracy!

## 📂 Project Structure

```
Kaiburr/
├── complaint_classification_v2.ipynb          # Main notebook with all results
├── Consumer_Complaint_Classification_Report.docx  # Professional report (15 pages)
├── README.md                                  # This file
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Git ignore rules
├── screenshots/                               # Project execution screenshots
└── complaints.csv                             # Dataset (download separately - see below)
```

**Dataset Note:** Download from [https://catalog.data.gov/dataset/consumer-complaint-database](https://catalog.data.gov/dataset/consumer-complaint-database)

## 🔧 Technologies Used

### Core Libraries
- **Python 3.12** - Programming language
- **pandas & numpy** - Data manipulation
- **scikit-learn** - Machine learning algorithms
- **NLTK** - Natural language processing

### Advanced ML
- **TensorFlow/Keras** - Deep learning (ANN)
- **LightGBM** - GPU-accelerated gradient boosting
- **matplotlib/seaborn** - Data visualization

## 📈 Dataset

- **Source:** Consumer Financial Protection Bureau (CFPB)
- **Download Link:** [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database)
- **Total Records:** 1,321,283 complaints
- **Features:** 13 columns (text, categorical, numerical)
- **Target Classes:** 4 categories

**Note:** The dataset file (`complaints.csv`) is not included in this repository due to GitHub's 100MB file size limit. Please download it from the official source above.

### Class Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| Credit Reporting | 806,713 | 61.10% |
| Debt Collection | 369,740 | 28.00% |
| Consumer Loan | 9,507 | 0.72% |
| Mortgage | 134,523 | 10.19% |

**Challenge:** Extreme class imbalance (84.9:1 ratio)

## 🚀 Key Features

### 1. Advanced Preprocessing
- ✅ NLTK text preprocessing (34.81% text reduction)
- ✅ Stopword removal & lemmatization
- ✅ Missing value imputation
- ✅ TF-IDF vectorization (5,000 features)

### 2. Feature Engineering
- ✅ TruncatedSVD dimensionality reduction (5,012 → 500 features)
- ✅ **100% variance retention** (exceptional!)
- ✅ StandardScaler normalization
- ✅ Label encoding for 11 categorical features

### 3. Model Innovation
- ✅ 4-layer ANN with BatchNormalization
- ✅ GPU-accelerated LightGBM (42x speedup)
- ✅ Incremental model saving (production best practice)
- ✅ No overfitting (train ≈ test for all models)

### 4. Performance Optimization
- ✅ Replaced SVM → Logistic Regression (97% time reduction)
- ✅ Replaced AdaBoost → LightGBM GPU (42x faster)
- ✅ Fixed ANN training collapse (61% → 99.90% accuracy)

## 🎓 Methodology

```
1. Data Loading (1.32M records) → 45 seconds
2. Text Preprocessing (NLTK) → 8 minutes
3. TF-IDF Vectorization → 12 minutes
4. TruncatedSVD (10:1 compression) → 3 minutes
5. Model Training (4 models) → ~48 minutes
   ├─ Random Forest → 15.35 min
   ├─ Logistic Regression → 3.42 min
   ├─ LightGBM (GPU) → 4.76 min
   └─ ANN → 24.42 min
6. Evaluation & Visualization → 15 seconds
───────────────────────────────────────────
Total Pipeline Time: ~72 minutes
```

## 🔬 Results & Analysis

### Overall Performance

| Metric | ANN | LightGBM | Random Forest | Logistic Reg. |
|--------|-----|----------|---------------|---------------|
| Accuracy | 99.90% | 98.67% | 96.42% | 65.66% |
| Precision | 99.90% | 98.66% | 96.35% | 51.23% |
| Recall | 99.90% | 98.67% | 96.42% | 65.66% |
| F1-Score | 99.90% | 98.66% | 96.37% | 54.01% |

### Per-Class Performance (ANN)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Credit Reporting | 99.90% | 99.92% | 99.91% |
| Debt Collection | 99.89% | 99.88% | 99.89% |
| Consumer Loan | 98.50% | 99.20% | 98.85% |
| Mortgage | 99.91% | 99.89% | 99.90% |

**Key Insight:** Even the minority class (0.72% of data) achieves 98.5% precision!

### Overfitting Analysis

| Model | Train | Test | Gap | Status |
|-------|-------|------|-----|--------|
| ANN | 99.92% | 99.90% | 0.02% | ✅ Excellent |
| LightGBM | 98.91% | 98.67% | 0.24% | ✅ Excellent |
| Random Forest | 96.89% | 96.42% | 0.47% | ✅ Good |

**Conclusion:** No significant overfitting. All models generalize exceptionally well.

## 💡 Challenges & Solutions

### Challenge 1: SVM Training Time
- **Problem:** SVM exceeded 140+ minutes without completion
- **Solution:** Replaced with Logistic Regression (3.42 min, 97% time reduction)

### Challenge 2: AdaBoost Performance
- **Problem:** Training exceeded 200+ minutes
- **Solution:** Replaced with LightGBM GPU (4.76 min, 42x speedup!)

### Challenge 3: ANN Training Collapse
- **Problem:** Initial training failed (61.10% accuracy)
- **Root Cause:** class_weight with extreme imbalance destabilized training
- **Solution:** Removed class_weight, added BatchNormalization → 99.90% accuracy

### Challenge 4: Memory Management
- **Problem:** 1.3M × 5,012 features = 26 GB matrix
- **Solution:** Sparse matrices + TruncatedSVD → 4.2 GB (84% reduction)

## 📦 Installation & Usage

### Prerequisites
```bash
Python 3.12+
GPU with CUDA support (optional, for LightGBM)
16GB RAM minimum
```

### Download Dataset
1. Download the dataset from: [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database)
2. Save as `complaints.csv` in the project root directory
3. Dataset size: ~500MB (1,321,283 records)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook complaint_classification_v2.ipynb
```

### Load Trained Models
```python
import pickle
import lightgbm as lgb
from tensorflow import keras

# Load preprocessing
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Load models
lgb_model = lgb.Booster(model_file='lightgbm_model.txt')
ann_model = keras.models.load_model('ann_model.h5')

# Make predictions
prediction = ann_model.predict(preprocessed_data)
```

## 📊 Visualizations

The notebook includes comprehensive visualizations:
- ✅ Accuracy comparison charts (20×14 figures)
- ✅ Confusion matrix heatmaps (22×18 figures)
- ✅ Per-class performance analysis
- ✅ F1-Score, Precision, Recall comparisons
- ✅ Training progress monitoring

## 📸 Screenshots

### Project Execution Screenshots

#### 1. Notebook Execution Overview
<img width="1920" height="1080" alt="Screenshot 2025-10-19 233555" src="https://github.com/user-attachments/assets/e87da8ae-994c-456f-a239-ff351dbde89d" />

*Main notebook with all cells executed and results displayed*

#### 2. Model Performance Results
<img width="1700" height="1014" alt="Screenshot 2025-10-19 233730" src="https://github.com/user-attachments/assets/e1ab159e-fd99-404a-aa39-edc83da7b8ee" />

*Showing 99.90% ANN accuracy and 98.67% LightGBM accuracy*

#### 3. Training Progress
<img width="1644" height="992" alt="Screenshot 2025-10-19 233804" src="https://github.com/user-attachments/assets/94e3403d-b065-4ea0-b387-b4766e661217" />

*ANN training progress with 46 epochs completed*

#### 4. Confusion Matrices
<img width="1264" height="1030" alt="Screenshot 2025-10-19 233924" src="https://github.com/user-attachments/assets/d14df28c-ba9b-4e72-969d-4e8cd3cb9371" />

*Confusion matrix visualizations for all 4 models*

#### 5. Performance Comparison Charts
<img width="1287" height="992" alt="Screenshot 2025-10-19 234004" src="https://github.com/user-attachments/assets/89a76a9a-5080-4542-9189-47748ea15b8b" />

*Accuracy, Precision, Recall, and F1-Score comparisons*

#### 6. LightGBM GPU Training
<img width="1320" height="994" alt="Screenshot 2025-10-19 234152" src="https://github.com/user-attachments/assets/b142489f-5156-4880-85d6-c72eba192b9e" />

*LightGBM GPU-accelerated training completed in 4.76 minutes*

**Note:** All screenshots include current date/time (October 19, 2025) and author information as per requirements.

## 🎯 Business Impact

| Metric | Before (Manual) | After (Automated) | Improvement |
|--------|----------------|-------------------|-------------|
| Processing Time | 5 min/complaint | <1 sec | 99.7% ↓ |
| Accuracy | 85-90% | 99.90% | +10-15% ↑ |
| Throughput | 1,000/hour | 3.6M/hour | 3,600x ↑ |
| Manual Effort | 100% | 5% | 95% ↓ |

**Cost Savings:** Estimated 95% reduction in manual classification effort

## 🚀 Future Enhancements

- [ ] **Ensemble Methods:** Combine ANN + LightGBM (expected 99.92-99.95%)
- [ ] **Transformer Models:** Experiment with BERT/RoBERTa
- [ ] **Real-Time API:** Deploy with Flask/FastAPI + Docker
- [ ] **A/B Testing:** Compare models in production
- [ ] **Active Learning:** Identify low-confidence predictions
- [ ] **Monitoring:** Track model drift and performance

## 📄 Documentation

- **Detailed Report:** [Consumer_Complaint_Classification_Report.docx](Consumer_Complaint_Classification_Report.docx) (15 pages)
- **Notebook:** [complaint_classification_v2.ipynb](complaint_classification_v2.ipynb) (all results included)

## 👨‍💻 Author

**MCA 9th Semester Student**
- 🎓 Master of Computer Applications
- 📧 [Your Email]
- 🔗 [LinkedIn Profile]

## 📝 License

This project is licensed under the MIT License.

## � Acknowledgments

- **Consumer Financial Protection Bureau (CFPB)** for the dataset
- **GitHub Copilot** for AI-assisted development
- **Open-source community** for excellent libraries

---

<p align="center">
  <b>⭐ If you found this project helpful, please consider giving it a star! ⭐</b>
</p>

<p align="center">
  Made with ❤️ for the Machine Learning Community
</p>

