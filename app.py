import streamlit as st
from Cancer_Detection import detector

# Initialize detector
if not hasattr(st.session_state, 'detector'):
    data = detector.load_and_preprocess('data.csv')  # Update path
    detector.train_model(data)
    st.session_state.detector = detector

# Page config
st.set_page_config(
    page_title="Breast Cancer Detection",
    layout="centered"
)

st.title("Breast Cancer Diagnosis")
st.markdown("""
Enter tumor characteristics to assess malignancy risk.  
All measurements should be from diagnostic imaging.
""")

# Input form
with st.form("diagnosis_form"):
    cols = st.columns(3)
    inputs = {}
    
    # Organized by feature type with proper ranges
    feature_groups = {
        "Mean Features": detector.feature_columns[:10],
        "SE Features": detector.feature_columns[10:20],
        "Worst Features": detector.feature_columns[20:]
    }
    
    for group_name, features in feature_groups.items():
        with cols[list(feature_groups.keys()).index(group_name)]:
            st.subheader(group_name)
            for feature in features:
                params = st.session_state.detector.feature_ranges[feature]
                inputs[feature] = st.number_input(
                    label=feature.replace("_", " ").title(),
                    min_value=float(params['min']),
                    max_value=float(params['max']),
                    value=float(params['mean']),
                    step=0.0001 if "se" in feature else 0.01,
                    format="%.4f" if "se" in feature else "%.2f"
                )
    
    submitted = st.form_submit_button("Assess Diagnosis")

# Handle prediction
if submitted:
    try:
        diagnosis, confidence = st.session_state.detector.predict(inputs)
        
        st.divider()
        if diagnosis == "Cancer detected":
            st.error(f"## {diagnosis} ⚠️")
            st.progress(confidence)
            st.warning(f"Confidence: {confidence:.1%} - Immediate consultation recommended")
        else:
            st.success(f"## {diagnosis} ✅") 
            st.progress(1 - confidence)
            st.info(f"Confidence: {(1-confidence):.1%} - Routine follow-up advised")
        
        # Show interpretation
        st.markdown("""
        **Interpretation Guide:**
        - <90% confidence: Consider additional tests
        - 90-95%: Likely correct but verify
        - >95%: High confidence diagnosis
        """)
        
    except ValueError as e:
        st.error("Invalid input values detected")
        st.json(e.args[0])  # Show exact validation errors
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")