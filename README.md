# 🎵 Music Genre Classification using Machine Learning

![Project Banner](images/World-Music-Globe-.png)

## 🔍 Project Overview

This project aims to build a robust machine learning pipeline for classifying music into different genres based on extracted audio features from a Spotify dataset. The objective is to identify the most accurate model using techniques like logistic regression, decision trees, random forests, and XGBoost, while exploring model performance through EDA, feature engineering, and hyperparameter tuning.

This project was developed as part of the ADS 504 course final project at the University of San Diego.

---
## 📁 Project Structure
```plaintext
music-genre-classification_with_app/
│
├── colab_notebooks/          # Jupyter Notebooks for analysis and modeling
│   ├── 01_ExploratoryAnalysis_Feature_Engineering.ipynb
│   └── 02_Modeling.ipynb
│
├── app/                      # Streamlit Web Application
│   ├── streamlit_app.py     # Main application file
│   ├── data/                # Test data for the app
│   │   ├── X_test.csv
│   │   └── y_test.csv
│   └── model/               # Trained models and scalers
│       ├── best_xgboost_model.pkl
│       └── standard_scaler.pkl
│
├── data/                     # Original dataset
│   └── spotify_songs.csv    # Main dataset file
│
├── images/                   # All graphs and visualizations
│   ├── eda/                 # Exploratory data analysis plots
│   │   ├── correlation_matrix.png
│   │   ├── feature_distributions.png
│   │   ├── genre_distribution.png
│   │   └── radar_chart.html
│   └── models/              # Model performance and evaluation plots
│       ├── confusion_matrix.png
│       ├── feature_importances.png
│       ├── model_performance_comparison_bar_plot.png
│       ├── pca_visualization.png
│       └── roc_curve_ovr.png
│
├── reports/                  # Final report and presentation
│   ├── final_report.pdf
│   └── slides/
│       └── project_presentation.pdf
│
├── requirements.txt          # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🧠 Key Objectives

- Perform Exploratory Data Analysis (EDA) on Spotify audio features
- Engineer relevant features to improve classifier performance
- Train multiple classification models (Logistic Regression, Random Forest, XGBoost)
- Evaluate model performance and interpret results
- Visualize training accuracy and metrics
- **Deploy a user-friendly Streamlit web application for real-time predictions**

---

## 🚀 **NEW: Interactive Web Application**

We have implemented a **Streamlit web application** that allows users to:
- Input audio features for music genre classification
- Get real-time predictions using our trained XGBoost model
- View model confidence scores
- Experience the project in an interactive format

**To run the app locally:**
```bash
cd app
streamlit run streamlit_app.py
```

---

## 📊 Dataset

- **Name**: Spotify Music Features Dataset  
- **Source**: Kaggle / Spotify Web API  
- **Size**: ~11,000 records, 20+ audio-based features  
- **Format**: CSV, numeric features  
- **Target**: Music Genre (`genre` column)

---

## 🔧 Tools & Technologies

- **Python Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
- **Web Framework**: Streamlit for interactive application
- **Model Persistence**: Pickle for saving trained models
- **Google Colab**: For data analysis and model training (due to dataset size and resource needs)
- **Git/GitHub**: For collaboration and version control

---

## 📌 How to Run

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/music-genre-classification_with_app.git
cd music-genre-classification_with_app
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application:
```bash
cd app
streamlit run streamlit_app.py
```

### 4. Open the Jupyter notebooks (optional):
- Open `colab_notebooks/01_ExploratoryAnalysis_Feature_Engineering.ipynb` for data exploration
- Open `colab_notebooks/02_Modeling.ipynb` for model training and evaluation

---

## 🎯 **Key Features**

- **Comprehensive EDA**: Detailed analysis of audio features and genre distributions
- **Advanced Feature Engineering**: PCA, feature selection, and preprocessing
- **Multiple ML Models**: Comparison of Logistic Regression, Random Forest, and XGBoost
- **Interactive Web App**: User-friendly interface for real-time predictions
- **Model Persistence**: Saved models for easy deployment and reuse
- **Professional Documentation**: Complete project report and presentation

---

## 📃 License

This project is licensed under the MIT License.

## 🌐 Acknowledgments

- Spotify API for dataset access
- Scikit-learn documentation
- Kaggle contributors
- Streamlit for the web application framework



