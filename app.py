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

# --- CONFIGURAZIONE DELLA PAGINA E TITOLI ---
st.set_page_config(
    layout="centered",
    page_title="ML-CDS Prognostic Score Calculator",
    page_icon="🤖"
)

st.title('ML-CDS Prognostic Score Calculator')
st.markdown("""
Questa applicazione utilizza un modello di machine learning (XGBoost) per calcolare 
uno score di rischio personalizzato per pazienti con stenosi aortica. 
Inserisci i parametri del paziente nella barra laterale per ottenere lo score e 
un'analisi dettagliata dei fattori di rischio.
""")

# --- CARICAMENTO DEL MODELLO (CON CACHING PER EFFICIENZA) ---
@st.cache_resource
def load_model_and_explainer():
    """
    Carica il modello XGBoost e crea l'explainer SHAP.
    Questa funzione viene eseguita una sola volta grazie alla cache.
    """
    try:
        model = xgb.Booster()
        model.load_model('ML_CDS_final_model.json')
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except Exception as e:
        st.error(f"Errore critico nel caricamento del modello: {e}")
        st.error("Assicurati che il file 'ML_CDS_final_model.json' sia nel repository GitHub.")
        return None, None

model, explainer = load_model_and_explainer()


# --- DEFINIZIONE DELLE FEATURE ATTESE DAL MODELLO ---
# Questo elenco DEVE corrispondere esattamente all'ordine delle feature usate per il training.
EXPECTED_FEATURES_ORDER = [
    'age', 'sex', 'nyha', 'ckd', 'rhythm', 'TRgrade', 'MRgrade', 'LVMi',
    'LVEF', 'LAVI', 'ee_ratio', 'SVi', 'TAPSE', 'PAPs', 'RVFWS', 'LVGLS', 'PALS',
    'tapse_paps', 'rvfws_paps'
]


# --- INTERFACCIA UTENTE NELLA SIDEBAR ---
with st.sidebar:
    st.header("Parametri del Paziente")
    
    # Dati demografici e clinici
    age = st.slider('Età (anni)', 40, 100, 75, help="Età del paziente al momento della valutazione.")
    sex = st.radio('Sesso', ['Donna', 'Uomo'], help="Sesso biologico del paziente.")
    nyha = st.select_slider('Classe NYHA', options=[1, 2, 3, 4], value=2, help="Classe funzionale New York Heart Association.")
    
    st.subheader("Comorbidità")
    ckd = st.checkbox('Malattia Renale Cronica Severa (CKD)', help="Presenza di CKD severa (eGFR < 30 ml/min/1.73m²).")
    rhythm = st.checkbox('Fibrillazione Atriale', help="Presenza di fibrillazione atriale al basale.")

    st.subheader("Valori Ecocardiografici")
    lvef = st.slider('LVEF (%)', 20, 80, 55)
    lvgls = st.slider('LVGLS (valore assoluto, %)', 5.0, 25.0, 18.0, step=0.1)
    pals = st.slider('PALS (%)', 5.0, 40.0, 25.0, step=0.1)
    lavi = st.slider('LAVI (ml/m²)', 15, 80, 40)
    tapse = st.slider('TAPSE (mm)', 5, 35, 20)
    paps = st.slider('PAPs (mmHg)', 10, 100, 30)
    rvfws = st.slider('RVFWS (valore assoluto, %)', 5.0, 35.0, 22.0, step=0.1)
    ee_ratio = st.slider("E/e' ratio", 4.0, 30.0, 12.0, step=0.1)
    svi = st.slider("SVi (ml/m²)", 15, 60, 35)
    lvmi = st.slider("LVMi (g/m²)", 50, 200, 110)
    mrgrade = st.radio("Rigurgito Mitralico (MR)", ["Assente/Lieve (0)", "Moderato/Severo (1)"])
    trgrade = st.radio("Rigurgito Tricuspidale (TR)", ["Assente/Lieve (0)", "Moderato/Severo (1)"])


# --- LOGICA DEL BACKEND E VISUALIZZAZIONE DEI RISULTATI ---
if st.sidebar.button('**📈 Calcola Score di Rischio**', use_container_width=True):
    if model is not None and explainer is not None:
        
        # 1. Conversione degli input in un formato numerico
        input_data = {
            'age': float(age), 'sex': 1 if sex == 'Uomo' else 0, 'nyha': float(nyha),
            'ckd': 1 if ckd else 0, 'rhythm': 1 if rhythm else 0,
            'LVEF': float(lvef), 'LVGLS': float(lvgls), 'PALS': float(pals), 'LAVI': float(lavi),
            'TAPSE': float(tapse), 'PAPs': float(paps), 'RVFWS': float(rvfws),
            'ee_ratio': float(ee_ratio), 'SVi': float(svi), 'LVMi': float(lvmi),
            'MRgrade': 1 if "1" in mrgrade else 0,
            'TRgrade': 1 if "1" in trgrade else 0
        }

        # 2. Calcolo delle feature ingegnerizzate
        input_data['tapse_paps'] = (input_data['TAPSE'] / input_data['PAPs']) if input_data['PAPs'] > 0 else 0
        input_data['rvfws_paps'] = (input_data['RVFWS'] / input_data['PAPs']) if input_data['PAPs'] > 0 else 0
        
        # 3. Creazione del DataFrame, assicurando l'ordine corretto delle colonne
        input_df = pd.DataFrame([input_data])[EXPECTED_FEATURES_ORDER]

        # 4. Calcolo dello score e dei valori SHAP
        with st.spinner('Il modello sta analizzando i dati...'):
            dmatrix = xgb.DMatrix(input_df)
            score = model.predict(dmatrix)[0]
            shap_values = explainer(input_df)

        # 5. Visualizzazione dei risultati
        st.header("Risultati dell'Analisi")
        
        # Mostra lo score in modo prominente
        st.metric(label="Score Prognostico ML-CDS", value=f"{score:.3f}")
        
        # Fornisce un'interpretazione qualitativa dello score
        if score > df_final['ML_CDS'].quantile(0.75): # Esempio: se è nel quartile più alto
             st.error("🔴 Profilo di Rischio: Alto")
        elif score > df_final['ML_CDS'].median():
             st.warning("🟠 Profilo di Rischio: Medio-Alto")
        elif score > df_final['ML_CDS'].quantile(0.25):
             st.info("🟡 Profilo di Rischio: Medio-Basso")
        else:
             st.success("🟢 Profilo di Rischio: Basso")
        st.caption("L'interpretazione qualitativa si basa sui quartili del cohort di derivazione.")


        st.subheader("Spiegazione Personalizzata del Rischio (SHAP Analysis)")
        st.write("""
        Il grafico sottostante mostra come ogni parametro ha contribuito a formare lo score finale del paziente.
        - Le **barre rosse** rappresentano i fattori che hanno **aumentato** il rischio.
        - Le **barre blu** rappresentano i fattori che hanno **ridotto** il rischio.
        """)
        
        # Genera e mostra il grafico SHAP
        fig, ax = plt.subplots(figsize=(10, 12))
        shap.plots.waterfall(shap_values[0], max_display=len(input_df.columns), show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    else:
        st.error("Il modello non è stato caricato correttamente. Impossibile eseguire i calcoli.")
else:
    st.info("Inserisci i dati del paziente nella barra laterale e clicca su 'Calcola Score di Rischio' per iniziare.")