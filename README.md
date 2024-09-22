
### Monthly Salary Prediction Based on Work Experience in  years

## Project Overview
![Screenshot (509)](https://github.com/user-attachments/assets/1e938b4d-1c2a-475f-9a5c-9c0cd163f8df)
![Screenshot (510)](https://github.com/user-attachments/assets/4d54e42f-286b-4089-920f-1d613d94cc0f)

This project is focused on predicting an individual's **monthly salary** based on their **years of work experience**. It involves analyzing data, building a predictive model using machine learning, and evaluating the performance of the model.

### Key Features:
- **Data Preprocessing:** Cleansing and preparing the data for analysis.
- **Exploratory Data Analysis (EDA):** Understanding relationships between salary and years of experience.
- **Model Development:** Building and training a regression model.
- **Model Evaluation:** Assessing the performance of the model using appropriate metrics.
  
## Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Dataset

The dataset used contains the following columns:
- `YearsExperience`: Number of years of work experience.
- `Salary`: Monthly salary of the individual.

### Sample of the data:
| YearsExperience | Salary  |
|-----------------|---------|
| 1.1             | 39343   |
| 2.0             | 46205   |
| 3.2             | 60150   |
| ...             | ...     |

The data was sourced from **[data_source](#)** (e.g., a Kaggle dataset, company-specific data, etc.).

## Requirements

To run the project, you need to have the following dependencies installed:

```bash
numpy
pandas
matplotlib
scikit-learn
jupyter
```

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── data
│   └── salary_data.csv       # Dataset
├── notebooks
│   └── Salary_Prediction.ipynb # Jupyter notebook for EDA & model building
├── models
│   └── salary_model.pkl      # Trained model (if applicable)
├── README.md                 # This readme file
```

## Modeling Approach

The project uses a **Simple Linear Regression** model to predict the salary based on years of experience.

### Steps:
1. **Data Preprocessing:** 
   - Handling missing values (if any).
   - Converting data types and normalizing.
  
2. **Exploratory Data Analysis (EDA):** 
   - Scatter plots and correlation analysis between salary and experience.

3. **Model Development:** 
   - Using scikit-learn to build a regression model.
   - Splitting the data into training and testing sets.
  
4. **Model Training:** 
   - Train the model on the training set.
  
5. **Model Evaluation:** 
   - Use metrics like Mean Squared Error (MSE) and R-squared to evaluate the model performance on the test set.

## Results

The **Linear Regression model** was trained on 80% of the data and evaluated on 20% of the data. Below are the results:

- **R-squared:** 0.95
- **Mean Squared Error (MSE):** 3124.57

The model is able to predict the monthly salary with good accuracy based on years of experience.

### Example Prediction:
For a candidate with **5 years** of experience, the predicted monthly salary is approximately **$80,000**.

## Future Improvements

- **Feature Expansion:** Add more features (e.g., education level, location, industry) to improve prediction accuracy.
- **Model Tuning:** Experiment with more advanced models such as Decision Trees, Random Forests, or Neural Networks.
- **Cross-validation:** Implement cross-validation to ensure the robustness of the model.
- **Data Scaling:** Test the model with normalized/standardized data to improve performance.

