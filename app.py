# =============================================================================
# FILE: app.py
# AUTORE: Marco Penso
# DATA: 12/08/2025
# =============================================================================


import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE DELLA PAGINA ---
st.set_page_config(
    layout="wide",
    page_title="ML-CDS Aortic Stenosis Risk Calculator",
    page_icon="❤️"
)

# --- CARICAMENTO DEL MODELLO (CON CACHING PER EFFICIENZA) ---
@st.cache_resource
def load_resources():
    """Carica il modello XGBoost e crea l'explainer SHAP."""
    try:
        model = xgb.Booster()
        model.load_model('ML_CDS_final_model.json')
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except Exception as e:
        st.error(f"Errore critico nel caricamento del modello: {e}")
        st.error("Assicurarsi che il file 'ML_CDS_final_model.json' sia presente nel repository GitHub.")
        return None, None

model, explainer = load_resources()

# --- COSTANTI DELLO STUDIO (Valori dei quartili dal cohort di derivazione) ---
# Questi valori sono usati per classificare il rischio di un nuovo paziente.
QUARTILE_25_PERCENT = 0.5557
MEDIAN_SCORE = 0.9485
QUARTILE_75_PERCENT = 1.7101

# --- TITOLO E DESCRIZIONE ---
st.title('ML-CDS: A Machine Learning-Based Cardiac Damage Score for Aortic Stenosis')
st.markdown("This tool provides a personalized, continuous risk score for patients with moderate-to-severe aortic stenosis by integrating clinical and multi-chamber echocardiographic data.")
st.markdown("---")

# --- NUOVA SEZIONE: LEGENDA DELLE CLASSI DI RISCHIO (RISPONDE AL FEEDBACK) ---
with st.expander("ℹ️ Click here to see the Risk Score Legend and Interpretation", expanded=True):
    st.markdown(f"""
    The ML-CDS score is a continuous value where a higher number indicates a higher prognostic risk. 
    Patients are classified into four risk groups based on the score quartiles from the original study cohort, which allows for a standardized interpretation:

    - **<span style="color:green;">🟢 Low Risk</span>**: Score ≤ {QUARTILE_25_PERCENT}
    - **<span style="color:darkgoldenrod;">🟡 Medium-Low Risk</span>**: Score > {QUARTILE_25_PERCENT} and ≤ {MEDIAN_SCORE}
    - **<span style="color:orange;">🟠 Medium-High Risk</span>**: Score > {MEDIAN_SCORE} and ≤ {QUARTILE_75_PERCENT}
    - **<span style="color:red;">🔴 High Risk</span>**: Score > {QUARTILE_75_PERCENT}

    <small>_Note: A small change in an input parameter can sometimes shift the patient across a risk threshold (e.g., from 0.90 to 0.96), leading to a different risk classification. The waterfall plot below helps explain exactly why this happens._</small>
    """, unsafe_allow_html=True)


# --- LAYOUT A COLONNE PER INPUT E OUTPUT ---
input_col, output_col = st.columns([2, 1])

# --- COLONNA DEGLI INPUT ---
with input_col:
    st.header("Patient Data")

    with st.expander("**Patient-related factors**", expanded=True):
        col_age, col_sex, col_nyha = st.columns(3)
        age = col_age.number_input('Age (years)', min_value=18, max_value=110, value=75, step=1)
        sex = col_sex.selectbox('Biological Sex', ['Female', 'Male'])
        nyha = col_nyha.selectbox('NYHA Class', [1, 2, 3, 4])
        
        col_ckd, col_rhythm = st.columns(2)
        ckd = col_ckd.toggle('Severe Chronic Kidney Disease', help="eGFR < 30 ml/min/1.73m²")
        rhythm = col_rhythm.toggle('Atrial Fibrillation')

    with st.expander("**Cardiac-related factors (Echocardiography)**", expanded=True):
        st.subheader("Left Ventricle & Atrium")
        c1, c2, c3 = st.columns(3)
        lvef = c1.number_input('LVEF (%)', 20, 80, 55)
        lvmi = c1.number_input("LVMi (g/m²)", 50, 250, 110)
        
        lvgls = c2.number_input('LVGLS (abs. value, %)', 5.0, 25.0, 18.0, step=0.1)
        svi = c2.number_input("SVi (ml/m²)", 15, 70, 35)
        
        lavi = c3.number_input('LAVI (ml/m²)', 15, 100, 40)
        pals = c3.number_input('PALS (%)', 0.0, 50.0, 25.0, step=0.1, help="Peak Atrial Longitudinal Strain")
        
        st.subheader("Diastolic Function")
        ee_ratio = st.number_input("E/e' ratio", 4.0, 50.0, 12.0, step=0.1)

        st.subheader("Right Ventricle & Pulmonary Circulation")
        c4, c5, c6 = st.columns(3)
        paps = c4.number_input('PAPs (mmHg)', 10, 120, 30)
        tapse = c5.number_input('TAPSE (mm)', 5, 40, 20)
        rvfws = c6.number_input('RVFWS (abs. value, %)', 5.0, 40.0, 22.0, step=0.1)

        st.subheader("Valvular Regurgitation")
        col_mr, col_tr = st.columns(2)
        mrgrade = col_mr.selectbox("Mitral Regurgitation", ["None/Mild (0)", "Moderate/Severe (1)"])
        trgrade = col_tr.selectbox("Tricuspid Regurgitation", ["None/Mild (0)", "Moderate/Severe (1)"])

# --- Pulsante di calcolo ---
calculate_button = input_col.button('**Calculate Risk Score & Generate Analysis**', type="primary", use_container_width=True)


# --- COLONNA DELL'OUTPUT ---
with output_col:
    st.header("Prognostic Assessment")
    
    if calculate_button:
        if model is not None and explainer is not None:
            # Crea il dataframe di input (logica invariata)
            input_data = {
                'age': age, 'sex': 1 if sex == 'Male' else 0, 'nyha': nyha,
                'ckd': 1 if ckd else 0, 'rhythm': 1 if rhythm else 0,
                'LVEF': lvef, 'LVGLS': lvgls, 'PALS': pals, 'LAVI': lavi,
                'TAPSE': tapse, 'PAPs': paps, 'RVFWS': rvfws, 'ee_ratio': ee_ratio,
                'SVi': svi, 'LVMi': lvmi, 'MRgrade': 1 if "1" in mrgrade else 0,
                'TRgrade': 1 if "1" in trgrade else 0
            }
            input_data['tapse_paps'] = (input_data['TAPSE'] / input_data['PAPs']) if input_data['PAPs'] > 0 else 0
            input_data['rvfws_paps'] = (input_data['RVFWS'] / input_data['PAPs']) if input_data['PAPs'] > 0 else 0
            
            expected_features_order = model.feature_names
            input_df = pd.DataFrame([input_data])[expected_features_order]

            with st.spinner('Analyzing patient data...'):
                dmatrix = xgb.DMatrix(input_df)
                score = model.predict(dmatrix)[0]
                shap_values = explainer(input_df)

            # Visualizzazione migliorata
            st.subheader("Patient's Risk Profile")
            col1, col2 = st.columns(2)
            col1.metric(label="Calculated ML-CDS Score", value=f"{score:.3f}")
            
            if score > QUARTILE_75_PERCENT:
                 risk_class = "High"
                 col2.error(f"**Risk Class:** {risk_class}", icon="🔴")
            elif score > MEDIAN_SCORE:
                 risk_class = "Medium-High"
                 col2.warning(f"**Risk Class:** {risk_class}", icon="🟠")
            elif score > QUARTILE_25_PERCENT:
                 risk_class = "Medium-Low"
                 col2.info(f"**Risk Class:** {risk_class}", icon="🟡")
            else:
                 risk_class = "Low"
                 col2.success(f"**Risk Class:** {risk_class}", icon="🟢")
            
            st.markdown("---")
            st.subheader("Individual Risk Factor Analysis (SHAP)")
            st.write("This plot explains **how the score was calculated**, showing how each parameter pushed the risk **up (red)** or **down (blue)** from the average.")
            
            fig, ax = plt.subplots(figsize=(8, 10))
            shap.plots.waterfall(shap_values[0], max_display=len(input_df.columns), show=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    else:
        st.info("Enter patient data and click the button below to see the results.")


# --- DISCLAIMER LEGALE ---
st.markdown("---")
st.subheader("Disclaimer")
st.markdown("""
<small> The ML-CDS Risk Calculator is owned by **Istituto Auxologico Italiano**.

Istituto Auxologico Italiano provides this webpage and calculator “as is” and in good faith as a tool free for unrestricted online use by patients, clinicians, and researchers. The calculator must not be used for any commercial use. The user assumes all responsibility for use of the calculator. Istituto Auxologico Italiano accepts no liability whatsoever for any harm, direct or indirect, real or perceived, loss, or damage resulting from its use or misuse. This tool is intended for informational purposes only and does not constitute medical advice.
</small>
""", unsafe_allow_html=True)

