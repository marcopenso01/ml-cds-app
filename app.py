# =============================================================================
# FILE: app.py
# DESCRIZIONE: Applicazione web Streamlit per il calcolo dello score ML-CDS.
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

# --- CARICAMENTO DELLE RISORSE ---
@st.cache_resource
def load_resources():
    try:
        model = xgb.Booster()
        model.load_model('ML_CDS_final_model.json')
        explainer = shap.TreeExplainer(model)
        df_ref = pd.read_excel('df_final.xlsx')
        return model, explainer, df_ref
    except Exception as e:
        st.error(f"Errore critico nel caricamento delle risorse: {e}")
        return None, None, None

model, explainer, df_final = load_resources()

# --- TITOLO E DESCRIZIONE ---
st.title('ML-CDS: A Machine Learning-Based Cardiac Damage Score for Aortic Stenosis')
st.markdown("This tool provides a personalized, continuous risk score for patients with moderate-to-severe aortic stenosis by integrating clinical and multi-chamber echocardiographic data.")
st.markdown("---")

# --- LAYOUT A COLONNE ---
input_col, output_col = st.columns([2, 1])

# --- COLONNA DEGLI INPUT ---
with input_col:
    st.header("Patient Data")

    with st.expander("**Patient-related factors**", expanded=True):
        col_age, col_sex, col_nyha = st.columns(3)
        age = col_age.number_input('Age (years)', 18, 110, 75)
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
        # --- MODIFICA CHIAVE: Aggiunto PALS qui ---
        pals = c3.number_input('PALS (%)', 0.0, 50.0, 25.0, step=0.1, help="Peak Atrial Longitudinal Strain")
        
        st.subheader("Diastolic Function")
        ee_ratio = st.number_input("E/e' ratio", 4.0, 40.0, 12.0, step=0.1)

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
calculate_button = input_col.button('**Calculate Risk Score**', type="primary", use_container_width=True)

# --- COLONNA DELL'OUTPUT ---
with output_col:
    st.header("Prognostic Assessment")
    
    if calculate_button:
        if model is not None and explainer is not None and df_final is not None:
            # Crea il dizionario con TUTTI i dati dall'interfaccia
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
            
            # Forza l'ordine corretto delle colonne
            expected_features_order = model.feature_names
            input_df = pd.DataFrame([input_data])[expected_features_order]

            # ... (resto della logica di calcolo e visualizzazione, che rimane invariata) ...
            with st.spinner('Analyzing patient data...'):
                dmatrix = xgb.DMatrix(input_df)
                score = model.predict(dmatrix)[0]
                shap_values = explainer(input_df)

            st.subheader("ML-CDS Score")
            st.markdown(f"<div style='background-color:#F0F2F6; padding: 15px; border-radius: 10px; text-align: center;'><h1 style='color: #262730;'>{score:.3f}</h1></div>", unsafe_allow_html=True)
            
            if score > df_final['ML_CDS'].quantile(0.75):
                 risk_class = "High Risk"
                 st.error(f"**Risk Class:** {risk_class}", icon="🔴")
            elif score > df_final['ML_CDS'].median():
                 risk_class = "Medium-High Risk"
                 st.warning(f"**Risk Class:** {risk_class}", icon="🟠")
            elif score > df_final['ML_CDS'].quantile(0.25):
                 risk_class = "Medium-Low Risk"
                 st.info(f"**Risk Class:** {risk_class}", icon="🟡")
            else:
                 risk_class = "Low Risk"
                 st.success(f"**Risk Class:** {risk_class}", icon="🟢")
            st.caption("Risk class is determined by the score's quartile within the original study cohort.")
            
            st.subheader("Individual Risk Factor Analysis")
            fig, ax = plt.subplots(figsize=(8, 10))
            shap.plots.waterfall(shap_values[0], max_display=len(input_df.columns), show=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.caption("This waterfall plot shows how each parameter contributed to the final score.")

    else:
        st.info("Enter patient data and click the 'Calculate Risk Score' button to see the results.")
        
# --- DISCLAIMER LEGALE IN FONDO ALLA PAGINA ---
st.markdown("---")
st.subheader("Disclaimer")
st.markdown("""
<small> The ML-CDS Risk Calculator is owned by **Istituto Auxologico Italiano**.

Istituto Auxologico Italiano provides this webpage and calculator “as is” and in good faith as a tool free for unrestricted online use by patients, clinicians, and researchers. The calculator must not be used for any commercial use. The user assumes all responsibility for use of the calculator. Istituto Auxologico Italiano accepts no liability whatsoever for any harm, direct or indirect, real or perceived, loss, or damage resulting from its use or misuse. This tool is intended for informational purposes only and does not constitute medical advice.
</small>
""", unsafe_allow_html=True)
