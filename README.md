# 🎵 Music Genre Classification using Machine Learning

![Project Banner](images/World-Music-Globe-.png)

## 🔍 Project Overview

This project aims to build a robust machine learning pipeline for classifying music into different genres based on extracted audio features from a Spotify dataset. The objective is to identify the most accurate model using techniques like logistic regression, decision trees, and random forests, while exploring model performance through EDA, feature engineering, and hyperparameter tuning.

This project was developed as part of the ADS 504 course final project at the University of San Diego.

---
## 📁 Project Structure
```plaintext
music-genre-classification/
│
├── colab_notebooks/ # Jupyter Notebooks hosted in Google Colab
│ ├── 01_Exploratory_Data_Analysis.ipynb
│ ├── 02_Feature_Engineering.ipynb
│ └── 03_Model_Training_and_Evaluation.ipynb
│
├── data/ # Placeholder for data files #⚠️ Please upload the dataset manually to your Google Colab session under the `data/` directory, as it is not included in the repository.

│
├── images/ # All graphs and visualizations
│ ├── eda/
│ └── models/
│
├── reports/ # Final report and presentation
│ ├── final_report.pdf
│ └── slides/
│ └── project_presentation.pdf
│
├── requirements.txt # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🧠 Key Objectives

- Perform Exploratory Data Analysis (EDA) on Spotify audio features
- Engineer relevant features to improve classifier performance
- Train multiple classification models
- Evaluate model performance and interpret results
- Visualize training accuracy and metrics

---

## 📊 Dataset

- **Name**: Spotify Music Features Dataset  
- **Source**: Kaggle / Spotify Web API  
- **Size**: ~11,000 records, 20+ audio-based features  
- **Format**: CSV, numeric features  
- **Target**: Music Genre (`genre` column)

---

## 🔧 Tools & Technologies

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Google Colab (due to dataset size and resource needs)
- Git/GitHub for collaboration and version control

---

## 📌 How to Run

1. Clone the repo:
```bash
   git clone https://github.com/your-username/music-genre-classification.git
```
2. Open the Colab notebooks:

Start with 01_Exploratory_Data_Analysis.ipynb on Google Colab

3. Install dependencies locally (optional):

```bash
pip install -r requirements.txt
```
📃 License
This project is licensed under the MIT License.

🌐 Acknowledgments
- Spotify API for dataset access

- Scikit-learn documentation

- Kaggle contributors  



