import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Breast Cancer Survival Prediction",
    page_icon="🎗️",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model_path = os.path.join(BASE_DIR, 'model', 'model.pkl')
    pipe_path = os.path.join(BASE_DIR, 'model', 'pipe.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(pipe_path, 'rb') as f:
        preprocessor = pickle.load(f)

    # Maintain backward compatibility across sklearn versions
    for _, trans in getattr(preprocessor, 'named_transformers_', {}).items():
        if hasattr(trans, 'named_steps'):
            for _, step in trans.named_steps.items():
                if hasattr(step, '_fit_dtype') and not hasattr(step, '_fill_dtype'):
                    step._fill_dtype = step._fit_dtype
                if not hasattr(trans, 'transform_input'):
                    trans.transform_input = None

    return model, preprocessor

model, preprocessor = load_artifacts()

def main():
    st.title("🎗️ Breast Cancer Survival Prediction")
    st.markdown(
        """
        This application predicts the patient survival outcome (**Alive** vs **Dead**)
        based on clinical features, protein expression levels, tumor stage, and surgical details.
        """
    )
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & Tumor Characteristics")
        Age = st.number_input("Age", min_value=18, max_value=120, value=50, step=1)
        Gender = st.selectbox("Gender", options=["Female", "Male"])
        Tumour_Stage = st.selectbox("Tumor Stage", options=["I", "II", "III"], index=1)
        Histology = st.selectbox(
            "Histology",
            options=[
                "Infiltrating Ductal Carcinoma",
                "Infiltrating Lobular Carcinoma",
                "Mucinous Carcinoma"
            ]
        )
        Surgery_type = st.selectbox(
            "Surgery Type",
            options=[
                "Other",
                "Lumpectomy",
                "Modified Radical Mastectomy",
                "Simple Mastectomy"
            ]
        )
        HER2_status = st.selectbox("HER2 Status", options=["Negative", "Positive"])

    with col2:
        st.subheader("Receptor Status & Protein Biomarkers")
        ER_status = st.selectbox("Estrogen Receptor (ER) Status", options=["Positive", "Negative"])
        PR_status = st.selectbox("Progesterone Receptor (PR) Status", options=["Positive", "Negative"])
        Protein1 = st.number_input("Protein 1 Expression", value=-0.08, format="%.5f")
        Protein2 = st.number_input("Protein 2 Expression", value=0.98, format="%.5f")
        Protein3 = st.number_input("Protein 3 Expression", value=-0.04, format="%.5f")
        Protein4 = st.number_input("Protein 4 Expression", value=0.08, format="%.5f")

    st.write("")
    if st.button("🔍 Predict Patient Survival", type="primary", use_container_width=True):
        input_df = pd.DataFrame({
            'Age': [Age],
            'Gender': [Gender.upper()],
            'Protein1': [Protein1],
            'Protein2': [Protein2],
            'Protein3': [Protein3],
            'Protein4': [Protein4],
            'Tumour_Stage': [Tumour_Stage],
            'Histology': [Histology],
            'HER2 status': [HER2_status],
            'Surgery_type': [Surgery_type],
            'ER status': [ER_status],
            'PR status': [PR_status]
        })

        try:
            processed_data = preprocessor.transform(input_df)
            raw_pred = model.predict(processed_data)
            prediction = int(np.ravel(raw_pred)[0])

            survival_labels = {
                0: "Alive (Positive Survival Outcome)",
                1: "Dead (Decreased Survival / High Risk)"
            }
            outcome_text = survival_labels.get(prediction, f"Class {prediction}")

            st.subheader("Prediction Result")
            if prediction == 0:
                st.success(f"### Result: **{outcome_text}**")
            else:
                st.warning(f"### Result: **{outcome_text}**")

            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(processed_data)[0]
                p_alive = probs[0] * 100
                p_dead = probs[1] * 100 if len(probs) > 1 else (100 - p_alive)
                st.write(f"- **Survival Probability (Alive):** `{p_alive:.2f}%`")
                st.write(f"- **Mortality Risk (Dead):** `{p_dead:.2f}%`")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
