# Student Exam Score Predictor

A comprehensive machine learning-based web application for predicting student academic performance using advanced analytics and personalized recommendations.

## Project Information

- **Developer:** Bichan Poudel
- **Program:** Bachelor of Computer Science and Information Technology (CSIT)
- **Semester:** 7th Semester
- **Roll Number:** 79010168
- **University:** Tribhuvan University (TU)
- **Project Type:** Final Year Project
- **Academic Year:** 2024-2025

## Overview

This application leverages machine learning algorithms to predict student examination scores by analyzing a comprehensive set of academic, lifestyle, and personal factors. The system provides not only accurate score predictions but also personalized improvement recommendations to help students optimize their academic performance.

### Key Features

- **Intelligent Score Prediction**: Machine learning-powered exam score forecasting
- **GPA Conversion**: Automatic conversion to 4.0 scale GPA with letter grades
- **Personalized Recommendations**: Data-driven suggestions for performance improvement
- **Comprehensive Analytics**: Detailed performance metrics and insights
- **User-Friendly Interface**: Intuitive web-based application built with Streamlit

## Input Parameters

The prediction model analyzes the following factors:

### Academic Factors
- Study hours per day
- Class attendance percentage
- Parental education level

### Health & Lifestyle
- Daily sleep hours
- Weekly exercise frequency
- Mental health rating (1-10 scale)

### Digital Habits
- Daily social media usage
- Daily entertainment consumption
- Internet connection quality

### Personal Factors
- Age
- Gender
- Part-time job status
- Extracurricular participation
- Dietary quality

## Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| Algorithm | Linear Regression | Machine learning algorithm used |
| Training Samples | 727 | Number of student records used for training |
| Test Samples | 182 | Number of student records used for validation |
| Root Mean Squared Error (RMSE) | 5.48 | Average prediction error in percentage points |
| R² Score | 0.889 | Coefficient of determination (88.9% accuracy) |
| Model Accuracy | 88.9% | Overall prediction accuracy |

## Technical Architecture

### Machine Learning Pipeline
1. **Data Preprocessing**: Feature engineering and encoding
2. **Model Training**: Linear regression with cross-validation
3. **Feature Selection**: 18 optimized input features
4. **Model Evaluation**: Comprehensive performance metrics
5. **Deployment**: Web application with real-time predictions

### Technology Stack
- **Programming Language**: Python 3.8+
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Web Framework**: Streamlit
- **Data Visualization**: Matplotlib, Seaborn
- **Model Serialization**: Joblib

## Project Structure

```
student-exam-predictor/
├── app.py                          # Main Streamlit application
├── notebook.ipynb                  # Data analysis and model development
├── best_model.pkl                  # Trained machine learning model
├── student_habits_performance.csv  # Training dataset
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
└── .git/                           # Version control
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Local Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/bichanpoudel/student-exam-predictor.git
   cd student-exam-predictor
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

5. **Access the Application**
   - Open your web browser
   - Navigate to: `http://localhost:8501`

## Usage Guide

### Step-by-Step Instructions

1. **Access the Application**: Open the web interface in your browser
2. **Navigate to Score Prediction**: Click on the "Score Prediction" tab
3. **Enter Student Information**:
   - Fill in all required fields with accurate information
   - Use sliders and dropdowns for data input
4. **Generate Prediction**: Click the "GENERATE PREDICTION" button
5. **Review Results**:
   - View predicted exam score
   - Check GPA equivalent and letter grade
   - Read personalized improvement suggestions
   - Analyze performance metrics

### Input Validation
- All fields are required for accurate predictions
- Age range: 10-30 years
- Study hours: 0-12 hours per day
- Attendance: 0-100%
- Sleep hours: 0-12 hours per day

## Model Interpretation

### Score Ranges and Classifications
- **90-100**: Excellent Performance (A grade)
- **80-89**: Very Good Performance (B grade)
- **70-79**: Good Performance (C grade)
- **60-69**: Average Performance (D grade)
- **Below 60**: Below Average (F grade)

### GPA Conversion Scale
- A+ (97-100): 4.0 GPA
- A (93-96): 4.0 GPA
- A- (90-92): 3.7 GPA
- B+ (87-89): 3.3 GPA
- B (83-86): 3.0 GPA
- B- (80-82): 2.7 GPA
- C+ (77-79): 2.3 GPA
- C (73-76): 2.0 GPA
- C- (70-72): 1.7 GPA
- D+ (67-69): 1.3 GPA
- D (63-66): 1.0 GPA
- D- (60-62): 0.7 GPA
- F (Below 60): 0.0 GPA

## Deployment

### Streamlit Cloud Deployment
1. Push code to GitHub repository
2. Visit [Streamlit Cloud](https://share.streamlit.io)
3. Connect GitHub repository
4. Deploy the application
5. Access via provided URL

### Local Deployment
- Follow the installation steps above
- Run `streamlit run app.py`
- Access at `http://localhost:8501`

## Data Privacy & Ethics

### Privacy Protection
- No personal identification data is stored
- All predictions are processed locally
- Input data is not transmitted or saved
- User information remains confidential

### Ethical Considerations
- Model predictions are estimates, not guarantees
- Recommendations are suggestions, not mandates
- Users should consult academic advisors for important decisions
- System designed to support, not replace, human guidance

## Future Enhancements

### Planned Features
- Additional machine learning algorithms comparison
- Advanced visualization dashboards
- Mobile application development
- Integration with learning management systems
- Multi-language support

### Technical Improvements
- Deep learning model implementation
- Real-time data collection and model updating
- Advanced feature engineering
- Performance optimization
- Cloud-native architecture

## Contributing

### Development Guidelines
1. Fork the repository
2. Create a feature branch
3. Make changes with proper documentation
4. Test thoroughly
5. Submit pull request

### Code Standards
- Follow PEP 8 style guidelines
- Include docstrings for functions
- Write comprehensive unit tests
- Maintain code documentation

## License

This project is developed as part of an academic requirement for Tribhuvan University. All rights reserved.

## Acknowledgments

- **Supervisor**: [Supervisor Name]
- **Department**: Computer Science and Information Technology
- **University**: Tribhuvan University
- **Dataset Source**: Student performance data collected for research purposes

## Contact Information

- **Developer**: Bichan Poudel
- **Email**: [Your email address]
- **LinkedIn**: [Your LinkedIn profile]
- **GitHub**: [Your GitHub username]

---

**Note**: This application is developed for educational and research purposes. For production use, additional validation and security measures should be implemented.
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
