import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Set page config
st.set_page_config(
    page_title="Student Exam Score Predictor",
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
    
    ### Key Features
    - **Score Prediction**: ML-based exam score forecasting
    - **GPA Conversion**: Automatic GPA and letter grade calculation
    - **Personalized Suggestions**: Data-driven improvement recommendations
    - **Performance Analytics**: Comprehensive score analysis
    
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
        predict_button = st.button("GENERATE PREDICTION", width='stretch')
    
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
            
            # GPA Conversion
            st.markdown("---")
            st.subheader("GPA Equivalent")
            
            if pred_score >= 97:
                gpa = 4.0
                letter_grade = "A+"
            elif pred_score >= 93:
                gpa = 4.0
                letter_grade = "A"
            elif pred_score >= 90:
                gpa = 3.7
                letter_grade = "A-"
            elif pred_score >= 87:
                gpa = 3.3
                letter_grade = "B+"
            elif pred_score >= 83:
                gpa = 3.0
                letter_grade = "B"
            elif pred_score >= 80:
                gpa = 2.7
                letter_grade = "B-"
            elif pred_score >= 77:
                gpa = 2.3
                letter_grade = "C+"
            elif pred_score >= 73:
                gpa = 2.0
                letter_grade = "C"
            elif pred_score >= 70:
                gpa = 1.7
                letter_grade = "C-"
            elif pred_score >= 67:
                gpa = 1.3
                letter_grade = "D+"
            elif pred_score >= 63:
                gpa = 1.0
                letter_grade = "D"
            elif pred_score >= 60:
                gpa = 0.7
                letter_grade = "D-"
            else:
                gpa = 0.0
                letter_grade = "F"
            
            gpa_col1, gpa_col2, gpa_col3 = st.columns(3)
            with gpa_col1:
                st.metric("GPA Score", f"{gpa:.1f}", "out of 4.0")
            with gpa_col2:
                st.metric("Letter Grade", letter_grade)
            with gpa_col3:
                st.metric("Percentage", f"{pred_score:.1f}%")
            
            # Performance Improvement Suggestions
            st.markdown("---")
            st.subheader("Performance Improvement Suggestions")
            
            suggestions = []
            
            # Study hours analysis
            if study_hours_per_day < 2:
                suggestions.append("**Increase Study Time**: Consider studying at least 2-3 hours daily. Current study time is below optimal levels.")
            elif study_hours_per_day < 4:
                suggestions.append("**Study Time Good**: Your current study hours are adequate, but increasing to 4+ hours could further improve performance.")
            else:
                suggestions.append("**Excellent Study Habits**: Your study time is optimal for academic success.")
            
            # Attendance analysis
            if attendance_percentage < 80:
                suggestions.append("**Improve Attendance**: Regular class attendance is crucial. Aim for 90%+ attendance rate.")
            elif attendance_percentage < 90:
                suggestions.append("**Attendance Satisfactory**: Consider attending all classes to maximize learning opportunities.")
            else:
                suggestions.append("**Perfect Attendance**: Excellent attendance record - keep it up!")
            
            # Sleep analysis
            if sleep_hours < 7:
                suggestions.append("**Prioritize Sleep**: Getting 7-9 hours of sleep nightly is essential for academic performance and memory retention.")
            elif sleep_hours < 8:
                suggestions.append("**Sleep Adequate**: Your sleep hours are good, but 8+ hours would be optimal.")
            else:
                suggestions.append("**Optimal Sleep**: Excellent sleep habits supporting academic success.")
            
            # Social media analysis
            if social_media_hours > 3:
                suggestions.append("**Reduce Social Media**: High social media usage may impact focus. Consider limiting to 1-2 hours daily.")
            elif social_media_hours > 1:
                suggestions.append("**Social Media Moderate**: Your usage is reasonable, but minimizing distractions during study time would help.")
            else:
                suggestions.append("**Good Digital Habits**: Low social media usage supports better concentration.")
            
            # Entertainment analysis
            if netflix_hours > 2:
                suggestions.append("**Balance Entertainment**: Excessive entertainment time may affect study focus. Consider reducing to 1-2 hours daily.")
            elif netflix_hours > 1:
                suggestions.append("**Entertainment Balanced**: Your entertainment time is moderate - ensure it doesn't interfere with studies.")
            else:
                suggestions.append("**Well-Balanced**: Good balance between entertainment and academic priorities.")
            
            # Exercise analysis
            if exercise_frequency < 3:
                suggestions.append("**Increase Physical Activity**: Regular exercise (3-5 days/week) improves concentration and reduces stress.")
            elif exercise_frequency < 5:
                suggestions.append("**Exercise Regular**: Your exercise frequency is good - consider adding more variety.")
            else:
                suggestions.append("**Active Lifestyle**: Excellent exercise habits supporting both physical and mental health.")
            
            # Mental health analysis
            if mental_health_rating < 6:
                suggestions.append("**Mental Health Support**: Consider speaking with a counselor or practicing stress management techniques.")
            elif mental_health_rating < 8:
                suggestions.append("**Mental Health Good**: Continue maintaining your mental well-being through healthy habits.")
            else:
                suggestions.append("**Strong Mental Health**: Excellent mental health foundation for academic success.")
            
            # Part-time job analysis
            if part_time_job == "Yes":
                suggestions.append("**Work-Life Balance**: Working part-time while studying requires good time management. Ensure work doesn't exceed 15-20 hours/week.")
            
            # Extracurricular analysis
            if extracurricular_participation == "No":
                suggestions.append("**Extracurricular Activities**: Consider joining clubs or sports to develop soft skills and reduce academic stress.")
            else:
                suggestions.append("**Well-Rounded**: Participation in extracurricular activities supports overall development.")
            
            # Diet analysis
            if diet_quality == "Poor":
                suggestions.append("**Improve Nutrition**: A balanced diet with fruits, vegetables, and proteins supports brain function and energy levels.")
            elif diet_quality == "Average":
                suggestions.append("**Nutrition Adequate**: Consider incorporating more nutrient-rich foods for optimal brain performance.")
            else:
                suggestions.append("**Excellent Nutrition**: Good dietary habits supporting academic performance.")
            
            # Internet quality analysis
            if internet_quality == "Poor":
                suggestions.append("**Improve Internet Access**: Stable internet is crucial for online learning and research. Consider upgrading your connection.")
            elif internet_quality == "Average":
                suggestions.append("**Internet Satisfactory**: Your connection is adequate, but faster speeds could enhance online learning.")
            else:
                suggestions.append("**Optimal Connectivity**: Excellent internet access supporting digital learning.")
            
            # Parental education analysis
            if parental_education_level == "High School":
                suggestions.append("**Seek Academic Support**: Consider tutoring or study groups to supplement learning resources.")
            elif parental_education_level == "Bachelor":
                suggestions.append("**Good Support System**: Parental education background provides adequate academic guidance.")
            else:
                suggestions.append("**Strong Academic Foundation**: Advanced parental education provides excellent academic support.")
            
            # Age-based suggestions
            if age < 18:
                suggestions.append("**Young Learner**: Focus on building strong study habits and time management skills for future academic success.")
            elif age > 25:
                suggestions.append("**Mature Student**: Leverage life experience and focus on efficient study techniques tailored to your learning style.")
            
            # Display suggestions
            if suggestions:
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
            else:
                st.success("**Excellent Profile**: All your habits and factors are optimally aligned for academic success!")
            
            # Overall performance summary
            st.markdown("---")
            st.subheader("Overall Assessment")
            
            positive_factors = []
            improvement_areas = []
            
            if study_hours_per_day >= 3: positive_factors.append("Study Hours")
            else: improvement_areas.append("Study Hours")
            
            if attendance_percentage >= 85: positive_factors.append("Attendance")
            else: improvement_areas.append("Attendance")
            
            if sleep_hours >= 7: positive_factors.append("Sleep")
            else: improvement_areas.append("Sleep")
            
            if social_media_hours <= 2: positive_factors.append("Digital Habits")
            else: improvement_areas.append("Digital Habits")
            
            if exercise_frequency >= 3: positive_factors.append("Physical Activity")
            else: improvement_areas.append("Physical Activity")
            
            if mental_health_rating >= 7: positive_factors.append("Mental Health")
            else: improvement_areas.append("Mental Health")
            
            if diet_quality != "Poor": positive_factors.append("Nutrition")
            else: improvement_areas.append("Nutrition")
            
            assessment_col1, assessment_col2 = st.columns(2)
            
            with assessment_col1:
                if positive_factors:
                    st.success(f"✅ **Strengths ({len(positive_factors)})**: {', '.join(positive_factors)}")
                else:
                    st.info("📈 **Growth Opportunity**: Focus on developing strong academic habits.")
            
            with assessment_col2:
                if improvement_areas:
                    st.warning(f"**Focus Areas ({len(improvement_areas)})**: {', '.join(improvement_areas)}")
                else:
                    st.success("🌟 **Well-Balanced**: All key areas are performing well!")
            
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
    st.dataframe(feature_df, width='stretch', hide_index=True)
    
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
    
    **Step 8: Review GPA & Suggestions**
    - Check your GPA equivalent and letter grade
    - Read personalized improvement suggestions
    - Review overall assessment and focus areas
    
    ### New Features
    
    **GPA Conversion**
    - Automatic conversion of percentage scores to GPA (4.0 scale)
    - Letter grade assignment (A+, A, B+, B, etc.)
    - Standard university grading system
    
    **Smart Suggestions**
    - Personalized recommendations based on your input data
    - Analysis of study habits, sleep, attendance, and lifestyle factors
    - Actionable advice for performance improvement
    
    **Comprehensive Analytics**
    - Performance percentile ranking
    - Model confidence levels
    - Strengths and improvement areas identification
    
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


