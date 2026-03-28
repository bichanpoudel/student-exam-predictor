# Student Exam Score Predictor

A machine learning-based web application that predicts student exam scores based on academic, lifestyle, and personal factors.

## Project Information

- **Developer:** Bichan Poudel
- **Program:** CSIT (7th Semester)
- **Roll Number:** 79010168
- **University:** Tribhuvan University (TU)
- **Project Type:** Final Year Project

## Overview

This application uses a Linear Regression model to predict student exam scores. The predictor analyzes various factors including:

- **Academic Factors:** Study hours, attendance, parental education
- **Health & Lifestyle:** Sleep hours, exercise frequency, mental health rating
- **Digital Habits:** Social media usage, entertainment consumption, internet quality
- **Personal Factors:** Age, gender, part-time job status, extracurricular participation

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Linear Regression |
| Training Samples | 727 |
| Test Samples | 182 |
| RMSE | 5.48 |
| R² Score | 0.889 |
| Accuracy | 88.9% |

## Project Structure

```
project ml/
├── notebook.ipynb          # Data analysis and model training
├── student_habits_performance.csv  # Dataset
├── app.py                  # Streamlit web application
├── best_model.pkl          # Trained ML model
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- streamlit
- joblib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-exam-predictor.git
cd student-exam-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501/`

### Features

1. **Score Prediction Tab:**
   - Input student information through interactive forms
   - Get instant exam score predictions
   - View performance metrics and analysis

2. **Model Information Tab:**
   - Detailed model architecture details
   - Performance metrics visualization
   - Input features documentation

3. **Instructions Tab:**
   - Step-by-step user guide
   - Best practices for better predictions
   - Data privacy information

## Dataset

The model is trained on data from 909 students with 18 input features including:
- Demographics (age, gender)
- Academic performance (study hours, attendance)
- Health factors (sleep, exercise, mental health)
- Lifestyle (social media, entertainment)
- Educational background (parental education, internet quality)

## Data Processing

1. Missing values handling: Removed rows with missing parental education level
2. Categorical encoding: Used one-hot encoding with `drop_first=True`
3. Feature normalization: Features were used as-is for linear regression
4. Train-test split: 80% training, 20% testing

## Model Details

**Algorithm:** Linear Regression
- Simple, interpretable model
- Captures linear relationships between features and exam scores
- Fast training and prediction

**Prediction Range:** 0-100 (clamped to ensure valid exam scores)

## Future Improvements

- Feature scaling and normalization
- Advanced ensemble methods (Random Forest, XGBoost)
- Cross-validation for robust evaluation
- Feature importance analysis
- Hyperparameter tuning
- Deep learning approaches

## Disclaimer

This predictor is for educational purposes. Actual exam scores may vary based on many factors not captured in this model. The predictions should be used as a reference only.

## Technologies Used

- **Python:** Core programming language
- **Pandas & NumPy:** Data manipulation and analysis
- **Scikit-Learn:** Machine learning library
- **Streamlit:** Web application framework
- **Joblib:** Model serialization

## Author

**Bichan Poudel**
- CSIT Student, 7th Semester
- Tribhuvan University
- Roll No: 79010168

## License

This project is open-source and available for educational purposes.

## Contact

For questions or suggestions, please feel free to reach out.

---

**Last Updated:** March 28, 2026
