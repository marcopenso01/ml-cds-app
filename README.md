# ML-CDS: A Machine Learning-Based Cardiac Damage Score for Aortic Stenosis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://[INSERISCI-QUI-IL-TUO-URL-STREAMLIT])

## 📖 Overview

This repository contains the source code for the **ML-CDS (Machine Learning-Based Cardiac Damage Score)**, an interactive web application designed for the prognostic assessment of patients with moderate-to-severe aortic stenosis.

The tool implements a supervised machine learning model (XGBoost) trained on a comprehensive set of clinical and multi-chamber echocardiographic data. Unlike traditional categorical staging systems, the ML-CDS provides a **continuous, personalized risk score**, offering a more granular and accurate stratification of patient risk.

This application is intended as a companion tool for the research paper:
> *[Inserisci qui il titolo e la citazione del tuo paper una volta pubblicato]*

## ✨ Key Features

-   **Personalized Risk Score**: Calculates a continuous prognostic score (`ML_CDS`) based on individual patient parameters.
-   **Risk Stratification**: Classifies the patient into one of four risk quartiles (Low, Medium-Low, Medium-High, High) based on the derivation cohort.
-   **Explainable AI (XAI)**: Generates a SHAP waterfall plot to visualize how each individual feature contributes to the final risk score, providing transparent and clinically interpretable insights.
-   **Intuitive Interface**: A user-friendly web interface, inspired by clinical calculators like EuroSCORE II, allows for easy data entry and clear visualization of results.

## 🚀 Live Application

You can access and use the live application here:
**[https://[INSERISCI-QUI-IL-TUO-URL-STREAMLIT]](https://[INSERISCI-QUI-IL-TUO-URL-STREAMLIT])**

*(Nota: Sostituisci `[INSERISCI-QUI-IL-TUO-URL-STREAMLIT]` con il link che otterrai da Streamlit Community Cloud dopo il deploy.)*

## 🛠️ How to Use the App

1.  Navigate to the application URL.
2.  Enter the patient's demographic, clinical, and echocardiographic parameters in the input fields on the left sidebar.
3.  Click the "**Calculate Risk Score**" button.
4.  The results, including the numerical ML-CDS score, the risk class, and the SHAP waterfall plot, will be displayed in the main panel.

## 📁 Repository Structure

-   `app.py`: The main Streamlit application script.
-   `ML_CDS_final_model.json`: The pre-trained XGBoost model file.
-   `df_final.xlsx`: The reference dataset used to determine risk quartiles.
-   `requirements.txt`: A list of Python libraries required to run the application.

##  disclaimer

The ML-CDS Risk Calculator is owned by **Istituto Auxologico Italiano**.

This tool is provided "as is" and in good faith for unrestricted online use by clinicians and researchers. It is intended for informational and research purposes only and **does not constitute medical advice**. The user assumes all responsibility for the use of this calculator. Istituto Auxologico Italiano accepts no liability whatsoever for any harm, direct or indirect, real or perceived, loss, or damage resulting from its use or misuse.
