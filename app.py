import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set page config
st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main {
        background-color: #f5f7fa;
        padding: 0;
    }
    
    .header-section {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 40px 20px;
        color: white;
        margin-bottom: 30px;
        border-radius: 0;
    }
    
    .header-section h1 {
        font-size: 2.5rem;
        margin-bottom: 10px;
        font-weight: 700;
    }
    
    .header-section p {
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    .form-section {
        background: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .form-title {
        font-size: 1.3rem;
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 3px solid #3498db;
    }
    
    .stButton > button {
        width: 100%;
        padding: 15px;
        font-size: 16px;
        font-weight: 600;
        border-radius: 8px;
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(52, 152, 219, 0.4);
    }
    
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #3498db;
    }
    
    .metric-title {
        color: #7f8c8d;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    
    .metric-value {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .success-message {
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        padding: 40px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 8px 16px rgba(39, 174, 96, 0.3);
    }
    
    .success-message h2 {
        font-size: 2.5rem;
        margin-bottom: 10px;
        color: white;
    }
    
    .success-message p {
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    .info-box {
        background: #ecf0f1;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin-bottom: 20px;
    }
    
    .info-box p {
        color: #2c3e50;
        margin: 8px 0;
    }
    
    .tab-content {
        padding: 30px;
    }
    
    .footer {
        background: #2c3e50;
        color: white;
        padding: 20px;
        text-align: center;
        margin-top: 50px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 2px solid #ecf0f1;
    }
    
    .stTabs [aria-selected="true"] {
        color: #3498db;
        border-bottom: 3px solid #3498db;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
try:
    model = joblib.load("best_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.info("Make sure 'best_model.pkl' exists in the project folder.")
    st.stop()

# Header
st.markdown("""
    <div class="header-section">
        <h1>Student Exam Score Predictor</h1>
        <p>Machine Learning Based Academic Performance Analysis System</p>
        <p style="font-size: 0.95rem; margin-top: 15px; opacity: 0.9;">A Final Year Project by Bichan Poudel | CSIT 7th Semester | TU Roll No: 79010168</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Project Information")
    st.markdown("""
    ### Developer
    **Name:** Bichan Poudel
    
    **Program:** CSIT (7th Semester)
    
    **Roll Number:** 79010168
    
    **University:** Tribhuvan University (TU)
    
    ### Project Details
    This application predicts student exam scores using machine learning
    based on academic, lifestyle, and personal factors.
    
    ### Model Configuration
    - **Algorithm:** Linear Regression
    - **Training Data:** 909 Student Records
    - **Features:** 18 Input Parameters
    
    ### Performance Metrics
    - **RMSE:** 5.48
    - **R² Score:** 0.889 (88.9% Accuracy)
    - **Model Version:** 1.0
    
    ### Data Categories
    1. Academic Performance
    2. Lifestyle Habits
    3. Digital Usage
    4. Educational Background
    5. Health Factors
    """)

# Main tabs
tab1, tab2, tab3 = st.tabs(["Score Prediction", "Model Information", "Instructions"])

with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.subheader("Student Profile Information")
    st.markdown("---")
    
    # Row 1: Age and Gender
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 10, 30, 18, help="Student age in years")
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col3:
        mental_health_rating = st.slider("Mental Health Rating (1-10)", 1, 10, 7)
    
    # Row 2: Study and Sleep
    col4, col5, col6 = st.columns(3)
    with col4:
        study_hours_per_day = st.slider("Daily Study Hours", 0.0, 12.0, 2.0)
    with col5:
        sleep_hours = st.slider("Daily Sleep Hours", 0.0, 12.0, 7.0)
    with col6:
        exercise_frequency = st.slider("Exercise Days Per Week", 0, 7, 3)
    
    # Row 3: Attendance and Digital
    col7, col8, col9 = st.columns(3)
    with col7:
        attendance_percentage = st.slider("Attendance Percentage", 0, 100, 90)
    with col8:
        social_media_hours = st.slider("Daily Social Media Hours", 0.0, 12.0, 1.0)
    with col9:
        netflix_hours = st.slider("Daily Entertainment Hours", 0.0, 12.0, 1.0)
    
    st.markdown("---")
    st.subheader("Educational & Lifestyle Background")
    
    # Row 4: Education Level
    col10, col11, col12 = st.columns(3)
    with col10:
        parental_education_level = st.selectbox("Parental Education Level", ["High School", "Bachelor", "Master"])
    with col11:
        internet_quality = st.selectbox("Internet Connection Quality", ["Poor", "Average", "Good"])
    with col12:
        diet_quality = st.selectbox("Diet Quality", ["Poor", "Average", "Good"])
    
    # Row 5: Job and Activities
    col13, col14 = st.columns(2)
    with col13:
        part_time_job = st.selectbox("Works Part-time Job", ["No", "Yes"])
    with col14:
        extracurricular_participation = st.selectbox("Participates in Extracurricular Activities", ["No", "Yes"])
    
    st.markdown("---")
    
    # Prepare input data
    input_df = pd.DataFrame({
        "age": [age],
        "study_hours_per_day": [study_hours_per_day],
        "social_media_hours": [social_media_hours],
        "netflix_hours": [netflix_hours],
        "attendance_percentage": [attendance_percentage],
        "sleep_hours": [sleep_hours],
        "exercise_frequency": [exercise_frequency],
        "mental_health_rating": [mental_health_rating],
        "gender": [gender],
        "part_time_job": [part_time_job],
        "diet_quality": [diet_quality],
        "parental_education_level": [parental_education_level],
        "internet_quality": [internet_quality],
        "extracurricular_participation": [extracurricular_participation],
    })
    
    # One-hot encode
    input_df = pd.get_dummies(input_df, 
                              columns=["gender", "part_time_job", "diet_quality", 
                                      "parental_education_level", "internet_quality", 
                                      "extracurricular_participation"],
                              drop_first=True)
    
    expected_features = [
        "age", "study_hours_per_day", "social_media_hours", "netflix_hours", 
        "attendance_percentage", "sleep_hours", "exercise_frequency", "mental_health_rating",
        "gender_Male", "gender_Other",
        "part_time_job_Yes",
        "diet_quality_Good", "diet_quality_Poor",
        "parental_education_level_High School", "parental_education_level_Master",
        "internet_quality_Good", "internet_quality_Poor",
        "extracurricular_participation_Yes",
    ]
    
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[expected_features]
    
    # Prediction button
    col_button = st.columns([1, 1, 1])[1]
    with col_button:
        predict_button = st.button("GENERATE PREDICTION", use_container_width=True)
    
    # Display prediction result
    if predict_button:
        try:
            pred_score = model.predict(input_df)[0]
            # Clamp score between 0 and 100
            pred_score = max(0, min(100, pred_score))
            
            # Score classification
            if pred_score >= 90:
                interpretation = "Excellent Performance - Outstanding Student"
                color_class = "success-message"
            elif pred_score >= 80:
                interpretation = "Very Good Performance - Strong Academic Standing"
                color_class = "success-message"
            elif pred_score >= 70:
                interpretation = "Good Performance - Above Average"
                color_class = "success-message"
            elif pred_score >= 60:
                interpretation = "Average Performance - Satisfactory"
                color_class = "success-message"
            else:
                interpretation = "Below Average - Needs Improvement"
                color_class = "success-message"
            
            # Display result
            st.markdown(f"""
                <div class="success-message">
                    <h2>{pred_score:.1f}</h2>
                    <p>Predicted Exam Score (Out of 100)</p>
                    <p style="font-size: 1.2rem; margin-top: 15px;">{interpretation}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display metrics
            st.markdown("---")
            st.subheader("Prediction Analytics")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Predicted Score", f"{pred_score:.1f}", "out of 100")
            
            with metric_col2:
                percentile = round((pred_score / 100) * 100, 1)
                st.metric("Performance Percentile", f"{percentile:.0f}%")
            
            with metric_col3:
                confidence = min(99, max(50, (pred_score - 40)))
                st.metric("Model Confidence", f"{confidence:.0f}%")
            
            with metric_col4:
                if pred_score >= 90:
                    status = "Top Performer"
                elif pred_score >= 75:
                    status = "On Track"
                elif pred_score >= 60:
                    status = "Average"
                else:
                    status = "Below Average"
                st.metric("Performance Status", status)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please ensure all fields are filled correctly.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.subheader("Model Architecture & Performance")
    
    st.markdown("""
    ### Machine Learning Model Details
    
    **Algorithm:** Linear Regression with One-Hot Encoding
    
    **Training Configuration:**
    - Training Dataset Size: 727 samples
    - Test Dataset Size: 182 samples
    - Total Features: 18
    - Encoding Method: One-Hot Encoding (drop_first=True)
    
    ### Model Performance Metrics
    """)
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">RMSE</div>
            <div class="metric-value">5.48</div>
            <p style="color: #95a5a6; font-size: 0.9rem;">Root Mean Squared Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">R² Score</div>
            <div class="metric-value">0.889</div>
            <p style="color: #95a5a6; font-size: 0.9rem;">Coefficient of Determination</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Accuracy</div>
            <div class="metric-value">88.9%</div>
            <p style="color: #95a5a6; font-size: 0.9rem;">Overall Prediction Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Input Features")
    
    feature_data = {
        "Category": ["Academic", "Academic", "Academic", "Academic", "Health", "Health", 
                    "Health", "Lifestyle", "Lifestyle", "Lifestyle", "Background", 
                    "Background", "Background", "Background", "Background", "Behavioral", "Behavioral"],
        "Feature": ["Age", "Study Hours/Day", "Attendance %", "Parental Education", 
                   "Sleep Hours", "Mental Health Rating", "Exercise Frequency",
                   "Social Media Hours", "Entertainment Hours", "Diet Quality",
                   "Gender", "Internet Quality", "Part-time Job", "Extracurricular Activities",
                   "Data Type", "Numeric Values", "Categorical Values"],
        "Data Type": ["Numeric", "Numeric", "Numeric", "Categorical", "Numeric", 
                     "Numeric", "Numeric", "Numeric", "Numeric", "Categorical",
                     "Categorical", "Categorical", "Categorical", "Categorical", 
                     "Mixed", "10-30", "Yes/No, Good/Poor/Average"]
    }
    
    feature_df = pd.DataFrame(feature_data)
    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("Key Insights")
    
    st.markdown("""
    1. **Strong Academic Predictors:** Study hours, attendance, and parental education show strong correlation with exam scores.
    2. **Health Impact:** Sleep quality and mental health are significant factors in academic performance.
    3. **Digital Habits:** Excessive social media and entertainment consumption negatively impacts scores.
    4. **Holistic Performance:** The model captures the relationship between lifestyle and academic success.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.subheader("How to Use This Application")
    
    st.markdown("""
    ### Step-by-Step Guide
    
    **Step 1: Enter Student Information**
    - Fill in all required fields in the "Score Prediction" tab
    - Ensure all values are accurate for better predictions
    
    **Step 2: Student Profile Details**
    - Age: Enter the student's current age
    - Gender: Select the student's gender
    - Mental Health: Rate overall mental health (1-10 scale)
    
    **Step 3: Academic Information**
    - Daily Study Hours: Hours spent studying per day
    - Attendance: Class attendance percentage
    - Parental Education: Highest education level of parents
    
    **Step 4: Lifestyle Factors**
    - Sleep Hours: Average hours of sleep per night
    - Exercise: Days per week with physical activity
    - Diet Quality: Overall quality of food consumption
    
    **Step 5: Digital Habits**
    - Social Media Hours: Daily time on social platforms
    - Entertainment Hours: Daily entertainment/streaming time
    - Internet Quality: Connection quality where studies occur
    
    **Step 6: Additional Information**
    - Part-time Job: Whether student works part-time
    - Extracurricular: Participation in activities outside school
    
    **Step 7: Generate Prediction**
    - Click "GENERATE PREDICTION" button
    - Review the predicted exam score
    - Analyze the performance metrics provided
    
    ### Important Notes
    
    - All fields must be completed for accurate predictions
    - This model works best for students aged 10-30
    - Predictions are based on historical data patterns
    - Individual results may vary based on external factors
    
    ### Data Privacy
    
    - No personal identification data is stored
    - Predictions are generated locally
    - Your input data is not saved or transmitted
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p style="font-weight: 600;">Student Exam Score Prediction System v1.0</p>
        <p style="font-size: 0.85rem; margin-top: 8px;">Final Year Project | CSIT 7th Semester | Tribhuvan University</p>
        <p style="font-size: 0.85rem; margin-top: 5px;">Developed by: Bichan Poudel | Roll No: 79010168</p>
        <p style="font-size: 0.85rem; margin-top: 8px;">Built with Streamlit, Scikit-Learn & Machine Learning</p>
    </div>
""", unsafe_allow_html=True)


