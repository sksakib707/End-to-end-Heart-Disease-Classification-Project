#  End-to-End Heart Disease Classification Project

This project is an **end-to-end machine learning pipeline** built using **Scikit-learn** to predict the likelihood of heart disease based on patient health metrics.  

The workflow covers every stage of a real-world data science project - from data preprocessing and exploratory analysis to model selection, hyperparameter tuning, and performance evaluation.

---

##  Project Overview

The goal of this project is to build a **classification model** that can predict whether a person has heart disease or not based on a set of medical attributes.

The project demonstrates:
- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA) using Pandas and Matplotlib  
- Model training using multiple algorithms:
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Random Forest Classifier  
- Hyperparameter tuning using **GridSearchCV**  
- Model evaluation using metrics such as accuracy, precision, recall, and F1-score  

---

##  Tech Stack

- **Python 3**
- **Scikit-learn**
- **NumPy**
- **Pandas**
- **Matplotlib / Seaborn**
- **Jupyter Notebook**

---

##  Dataset

The dataset used is the **UCI Heart Disease Dataset**, a publicly available dataset often used for classification tasks.  
It contains various patient health indicators like age, cholesterol level, resting blood pressure, maximum heart rate, etc.

---

##  Model Performance

After training and hyperparameter tuning, the **Random Forest Classifier** achieved the best performance among the models tested.

---

##  Project Structure

heart-disease-project/
│
├── data/ # Dataset (if applicable)
├── notebooks/ # Jupyter notebooks
├── models/ # Trained models (optional)
├── README.md # Project description
├── .gitignore # Ignored files
└── main.ipynb # Main notebook


---

##  Installation and Setup

Follow these steps to run the project on your local machine:

### 1. Clone the repository
```bash
git clone https://github.com/sksakib707/End-to-end-Heart-Disease-Classification-Project.git
```
### 2. Navigate to the project folder
```bash
cd End-to-end-Heart-Disease-Classification-Project
```
### 3. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate  # For Windows
source venv/bin/activate  # For Mac/Linux
```
### 4. Install required dependencies
```bash
pip install -r requirements.txt
```
### 5. Run the notebook or Python script
```bash
jupyter notebook
```

Then open the project file (e.g. heart-disease-project.ipynb).

##  Future Improvements

- Try more advanced models (XGBoost, LightGBM, etc.)
- Add feature importance visualization
- Deploy as a web app using Streamlit or Flask

---

##  Author

**Sheikh Shadman Sakib**  
CSE Student | Machine Learning Enthusiast  
[GitHub](https://github.com/sksakib707) | [LinkedIn](https://www.linkedin.com/in/sheikh-shadman-sakib)

---

##  License

This project is for **educational purposes only**.
