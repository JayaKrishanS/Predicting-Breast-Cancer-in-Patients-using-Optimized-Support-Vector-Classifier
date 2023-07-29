import  pickle
import streamlit as st
from streamlit_option_menu import option_menu

model = pickle.load(open("Breast_cancer_model.jpg","rb"))

with st.sidebar:
    selected = option_menu(
        menu_title="Breast Cancer Prediction",
        options=["About project","---","Predict","---"],
        menu_icon="person-vcard",
        default_index=0,
        styles={"nav-link": {"--hover-color": "brown"}},
        orientation="vertical",
    )

if selected == "About project":
    st.title('Breast Cancer Prediction')
    st.markdown("<h5 style='color: orange;'>Predicting Breast Cancer in Patients using Optimized Support Vector Classifier </h5>", unsafe_allow_html=True)    
    st.markdown("* In this project, a dataset comprising approximately 569 instances was utilized. The project workflow involved several key steps to develop an accurate machine learning model.")
    st.markdown("* Firstly, the dataset was loaded and subjected to an exploratory data analysis (EDA) process. Following the EDA, a feature engineering process was conducted, which involved transforming and manipulating the dataset.")
    st.markdown("* To construct a machine learning model, an ensemble technique was employed, which consisted of building five different models: SVM classifier, logistic regression, random forest classifier, decision tree classifier, and GradientBoosting classifier. Each model brought unique strengths to the prediction task.")
    st.markdown("* However, the primary focus of the project was to optimize the performance of the SVM classifier through hyperparameter tuning. To achieve this, the dataset underwent feature scaling, a technique that normalizes the range of features. Subsequently, the SVM model was fine-tuned using grid search cross-validation (CV) to determine the optimal combination of hyperparameters.")
    st.markdown("* Upon obtaining the best hyperparameters, a new SVM classifier was trained and evaluated. The outcome of this tuning process yielded an impressive accuracy of 98%, showcasing a significant improvement compared to the initial accuracy of 94% obtained before hyperparameter tuning.")
    github_url_1 = "https://github.com/JayaKrishanS/Predicting-Breast-Cancer-in-Patients-using-Optimized-Support-Vector-Classifier.git"
    st.markdown(f'<a href="{github_url_1}" target="_blank"><i class="fab fa-github"></i>  GitHub</a>', unsafe_allow_html=True)
    st.markdown("---")

if (selected == 'Predict'):
    st.title('Breast Cancer Prediction')

    st.markdown("<h5 style='color: orange;'>Kindly enter the details for prediction</h5>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(' ')
        radius_mean = st.number_input("radius_mean", value=17.99)
        texture_mean = st.number_input("texture_mean", value= 10.38)
        perimeter_mean = st.number_input("perimeter_mean", value=122.8)
        area_mean = st.number_input("area_mean", value=1001.0)
        smoothness_mean = st.number_input("smoothness_mean", value=0.11840)
        compactness_mean = st.number_input("compactness_mean", value=0.27760)
        concavity_mean = st.number_input("concavity_mean", value=0.3001)
        concave_points_mean = st.number_input("concave points_mean", value=0.14710)
        symmetry_mean = st.number_input("symmetry_mean", value=0.2419)
        fractal_dimension_mean = st.number_input("fractal_dimension_mean", value=0.07871)
        radius_se = st.number_input("radius_se", value=1.0950)
        texture_se = st.number_input("texture_se", value=0.9053)
        perimeter_se = st.number_input("perimeter_se", value=8.589)
        area_se = st.number_input("area_se", value=153.40)
        smoothness_se = st.number_input("smoothness_se", value=0.006399)


    with col2:               
        compactness_se = st.number_input("compactness_se", value=0.04904)
        concavity_se = st.number_input("concavity_se", value=0.05373)
        concave_points_se = st.number_input("concave_points_se", value=0.01587)
        symmetry_se = st.number_input("symmetry_se", value=0.03003)
        fractal_dimension_se = st.number_input("fractal_dimension_se", value=0.006193)
        radius_worst = st.number_input("radius_worst", value=25.38)
        texture_worst = st.number_input("texture_worst", value=17.33)
        perimeter_worst = st.number_input("perimeter_worst", value=184.6)
        area_worst = st.number_input("area_worst", value=2019.0)
        smoothness_worst = st.number_input("smoothness_worst", value=0.1622)
        compactness_worst = st.number_input("compactness_worst", value=0.6656)
        concavity_worst = st.number_input("concavity_worst", value=0.7119)
        concave_points_worst = st.number_input("concave points_worst", value=0.2654)
        symmetry_worst = st.number_input("symmetry_worst", value=0.4601)
        fractal_dimension_worst = st.number_input("fractal_dimension_worst", value=0.11890)

status_predict = ''
if st.button('Predict'):
    status_prediction = model.predict([[radius_mean, texture_mean, perimeter_mean,area_mean, smoothness_mean, compactness_mean, concavity_mean,concave_points_mean, symmetry_mean, fractal_dimension_mean,radius_se, texture_se, perimeter_se, area_se, smoothness_se,compactness_se, concavity_se, concave_points_se, symmetry_se,fractal_dimension_se, radius_worst, texture_worst,perimeter_worst, area_worst, smoothness_worst,compactness_worst, concavity_worst, concave_points_worst,symmetry_worst, fractal_dimension_worst]])
    if (status_prediction[0] == 1):
        status_predict = 'The status is Malignant'
    else:
        status_predict = 'The status is Benign'
    
    st.success(status_predict)
