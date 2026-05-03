import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import base64
import json

st.set_page_config(page_title="PredictIQ — OCP", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background-color: #f5f0e8; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a4d2e 0%, #2d7a4f 50%, #1a4d2e 100%);
    padding: 20px 10px;
}
[data-testid="stSidebar"] * { color: #f0f7f0 !important; }
.kpi-card {
    background: white; border-radius: 18px; padding: 22px 20px;
    box-shadow: 0 4px 24px rgba(45,122,79,0.10); text-align: center;
    margin-bottom: 10px; border-top: 4px solid #2d7a4f;
}
.kpi-card-energy {
    background: white; border-radius: 18px; padding: 22px 20px;
    box-shadow: 0 4px 24px rgba(255,165,0,0.10); text-align: center;
    margin-bottom: 10px; border-top: 4px solid #f39c12;
}
.kpi-value { font-size: 2.1rem; font-weight: 700; color: #1a4d2e; }
.kpi-value-energy { font-size: 2.1rem; font-weight: 700; color: #e67e22; }
.kpi-label { font-size: 0.82rem; color: #888; margin-top: 4px; font-weight: 500; }
.kpi-sub   { font-size: 0.78rem; margin-top: 6px; font-weight: 600; }
.green  { color: #2d7a4f; }
.orange { color: #c8860a; }
.red    { color: #c0392b; }
.blue   { color: #1a4d2e; }
.page-header {
    background: linear-gradient(135deg, #1a4d2e 0%, #2d7a4f 100%);
    border-radius: 20px; padding: 28px 32px; margin-bottom: 24px;
}
.page-header-energy {
    background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%);
    border-radius: 20px; padding: 28px 32px; margin-bottom: 24px;
}
.page-header h1 { color: white !important; font-size: 1.8rem !important; margin: 0 !important; font-weight: 700 !important; }
.page-header-energy h1 { color: white !important; font-size: 1.8rem !important; margin: 0 !important; font-weight: 700 !important; }
.page-header p  { color: #a8d5b5 !important; margin: 6px 0 0 0 !important; font-size: 0.95rem !important; }
.page-header-energy p { color: #fdebd0 !important; margin: 6px 0 0 0 !important; font-size: 0.95rem !important; }
.alert-high { background: #fff5f5; border-left: 4px solid #c0392b; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; }
.alert-warn { background: #fffbf0; border-left: 4px solid #c8860a; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; }
.alert-ok   { background: #f0fff4; border-left: 4px solid #2d7a4f; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; }
.stButton > button {
    background: linear-gradient(135deg, #1a4d2e 0%, #2d7a4f 100%);
    color: white !important; border: none; border-radius: 12px;
    padding: 14px 30px; font-size: 16px; font-weight: 600; width: 100%;
    transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(45,122,79,0.3);
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(45,122,79,0.45); }
.stNumberInput input { border-radius: 10px !important; border: 1.5px solid #d4c9b0 !important; background: #faf7f2 !important; }
.stTextInput input { border-radius: 12px !important; border: 2px solid #d4c9b0 !important; background: white !important; padding: 12px 16px !important; }
h1 { color: #1a4d2e !important; font-weight: 700 !important; }
h2 { color: #1a4d2e !important; font-weight: 600 !important; }
h3 { color: #2d7a4f !important; font-weight: 600 !important; font-size: 1.1rem !important; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.badge-green  { background: #e8f5e9; color: #1a4d2e; }
.badge-orange { background: #fff8e1; color: #c8860a; }
.badge-red    { background: #ffebee; color: #c0392b; }
.progress-bar { background: #e8e0d0; border-radius: 10px; height: 8px; margin-top: 6px; overflow: hidden; }
.progress-fill { height: 100%; border-radius: 10px; background: linear-gradient(90deg, #1a4d2e, #2d7a4f); }
hr { border: none; border-top: 1.5px solid #e8e0d0; margin: 18px 0; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Chargement modeles ──────────────────────────────────
MODEL_PATH   = os.path.join('..', 'models', 'rf_model.pkl')
SCALER_PATH  = os.path.join('..', 'models', 'scaler.pkl')
ENERGIE_PATH = os.path.join('..', 'models', 'rf_energie.pkl')
SCALER_E_PATH= os.path.join('..', 'models', 'scaler_energie.pkl')

@st.cache_resource
def load_models():
    try:
        rf = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        sc = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        re = joblib.load(ENERGIE_PATH) if os.path.exists(ENERGIE_PATH) else None
        se = joblib.load(SCALER_E_PATH) if os.path.exists(SCALER_E_PATH) else None
        return rf, sc, re, se
    except Exception:
        return None, None, None, None

rf_model, scaler, rf_energie, scaler_energie = load_models()

# ── Recommandations energie ─────────────────────────────
RECO_PATH = os.path.join('..', 'models', 'energie_recommandations.json')
if os.path.exists(RECO_PATH):
    with open(RECO_PATH, 'r') as f:
        reco = json.load(f)
else:
    reco = {'heure_pointe': 9, 'heure_creuse': 6, 'n_anomalies': 1752,
            'economies_kwh': 82264.36, 'economies_mad': 98717.24,
            'conso_normale': 3.17, 'conso_anomalie': 50.0, 'prix_kwh_mad': 1.2}

# ── Logo ───────────────────────────────────────────────
try:
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo.jpeg')
except Exception:
    logo_path = 'logo.jpeg'

if os.path.exists(logo_path):
    with open(logo_path, 'rb') as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_html_sm = '<img src="data:image/jpeg;base64,' + logo_b64 + '" style="width:100px;height:100px;border-radius:50%;object-fit:cover;box-shadow:0 4px 16px rgba(0,0,0,0.2);margin-bottom:12px;"/>'
else:
    logo_html_sm = '<div style="font-size:3rem;margin-bottom:12px;">OCP</div>'

# ── Session State ──────────────────────────────────────
if "module" not in st.session_state:
    st.session_state.module = None
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour — Je suis le chatbot PredictIQ. Comment puis-je vous aider ?"}
    ]
if "messages_e" not in st.session_state:
    st.session_state.messages_e = [
        {"role": "assistant", "content": "Bonjour — Je suis le chatbot Energie PredictIQ. Comment puis-je vous aider ?"}
    ]

# ══════════════════════════════════════════════════════
# PAGE DE GARDE — 2 BOUTONS
# ══════════════════════════════════════════════════════
if st.session_state.module is None:

    st.markdown(
        '<div style="min-height:90vh;display:flex;flex-direction:column;justify-content:center;align-items:center;padding:40px 20px;">'
        '<div style="text-align:center;margin-bottom:40px;">'
        '<div style="display:inline-block;background:white;border-radius:20px;padding:16px 40px;box-shadow:0 4px 24px rgba(45,122,79,0.12);border-bottom:3px solid #2d7a4f;">'
        '<div style="font-size:0.8rem;color:#2d7a4f;font-weight:700;letter-spacing:3px;text-transform:uppercase;">Groupe OCP — OCP Phosboucraa</div>'
        '</div></div>'
        '<div style="background:linear-gradient(135deg,#1a4d2e 0%,#2d7a4f 60%,#3d9b6a 100%);border-radius:32px;padding:60px 80px;max-width:850px;width:100%;box-shadow:0 30px 80px rgba(26,77,46,0.35);text-align:center;">'
        '<div style="font-size:3.5rem;margin-bottom:20px;">OCP</div>'
        '<div style="font-size:3.8rem;font-weight:900;color:white;letter-spacing:4px;margin-bottom:8px;">PredictIQ</div>'
        '<div style="font-size:1rem;color:#c8f0d8;font-weight:500;letter-spacing:1px;margin-bottom:30px;">Intelligent Industrial AI Platform</div>'
        '<div style="width:80px;height:3px;background:rgba(255,255,255,0.4);border-radius:2px;margin:0 auto 30px auto;"></div>'
        '<div style="font-size:0.9rem;color:#a8d5b5;line-height:1.9;margin-bottom:40px;">'
        'Machine Learning &nbsp;•&nbsp; Deep Learning LSTM &nbsp;•&nbsp; Explainable AI<br>'
        'Maintenance Predictive &nbsp;•&nbsp; Optimisation Energetique'
        '</div>'
        '<div style="display:flex;justify-content:center;gap:30px;margin-bottom:40px;flex-wrap:wrap;">'
        '<div style="text-align:center;"><div style="font-size:1.6rem;font-weight:800;color:white;">0.9991</div><div style="font-size:0.75rem;color:#a8d5b5;margin-top:2px;">R2 Energie</div></div>'
        '<div style="width:1px;background:rgba(255,255,255,0.2);"></div>'
        '<div style="text-align:center;"><div style="font-size:1.6rem;font-weight:800;color:white;">0.8613</div><div style="font-size:0.75rem;color:#a8d5b5;margin-top:2px;">R2 LSTM</div></div>'
        '<div style="width:1px;background:rgba(255,255,255,0.2);"></div>'
        '<div style="text-align:center;"><div style="font-size:1.6rem;font-weight:800;color:white;">98 717</div><div style="font-size:0.75rem;color:#a8d5b5;margin-top:2px;">MAD economies</div></div>'
        '<div style="width:1px;background:rgba(255,255,255,0.2);"></div>'
        '<div style="text-align:center;"><div style="font-size:1.6rem;font-weight:800;color:white;">35 040</div><div style="font-size:0.75rem;color:#a8d5b5;margin-top:2px;">Observations</div></div>'
        '</div>'
        '<div style="background:rgba(255,255,255,0.1);border-radius:12px;padding:12px 24px;display:inline-block;">'
        '<div style="color:#c8f0d8;font-size:0.85rem;font-weight:500;">NASA CMAPSS FD001 &nbsp;|&nbsp; Steel Industry Energy Dataset</div>'
        '</div>'
        '</div>'
        '<div style="margin-top:28px;text-align:center;">'
        '<div style="font-size:0.9rem;color:#2d7a4f;font-style:italic;font-weight:500;">Anticipate. Explain. Prevent. Optimize.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])
    with col2:
        if st.button("Maintenance Predictive", key="btn_maintenance"):
            st.session_state.module = "maintenance"
            st.rerun()
    with col4:
        if st.button("Optimisation Energetique", key="btn_energie"):
            st.session_state.module = "energie"
            st.rerun()

# ══════════════════════════════════════════════════════
# MODULE MAINTENANCE PREDICTIVE
# ══════════════════════════════════════════════════════
elif st.session_state.module == "maintenance":

    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:10px 0 24px 0;">'
            + logo_html_sm +
            '<div style="font-size:1.3rem;font-weight:800;color:white;letter-spacing:1px;">PredictIQ</div>'
            '<div style="font-size:0.7rem;color:#a8d5b5;margin-top:2px;font-style:italic;">Maintenance Predictive</div>'
            '</div>', unsafe_allow_html=True)

        st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:0 0 16px 0;'>", unsafe_allow_html=True)

        st.markdown("""
        <div style='padding:12px 16px;background:rgba(255,255,255,0.10);border-radius:12px;margin-bottom:20px;'>
            <div style='font-size:0.75rem;color:#a8d5b5;'>Groupe OCP</div>
            <div style='font-size:0.95rem;font-weight:600;color:white;margin-top:2px;'>OCP Laayoune</div>
            <div style='font-size:0.75rem;color:#a8d5b5;margin-top:4px;'>Mai 2026</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio("Navigation", [
            "Dashboard", "Statistiques Avancees", "Prediction RUL",
            "SHAP Explainability", "LSTM Deep Learning", "Chatbot"
        ], key="nav_maintenance")

        st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:16px 0;'>", unsafe_allow_html=True)

        if st.button("Accueil", key="btn_accueil_m"):
            st.session_state.module = None
            st.rerun()

        statut = "Modele charge" if rf_model is not None else "Modele non charge"
        couleur = "#a8d5b5" if rf_model is not None else "#c0392b"
        st.markdown(
            '<div style="padding:12px;background:rgba(255,255,255,0.12);border-radius:12px;text-align:center;margin-top:12px;">'
            '<div style="color:' + couleur + ';font-weight:600;font-size:0.85rem;">' + statut + '</div>'
            '<div style="color:#c8dfc8;font-size:0.75rem;margin-top:4px;">NASA CMAPSS FD001</div>'
            '</div>', unsafe_allow_html=True)

    if page == "Dashboard":
        st.markdown("""<div class="page-header">
            <h1>Tableau de Bord — Maintenance Predictive</h1>
            <p>Groupe OCP — Surveillance intelligente des equipements — NASA CMAPSS FD001</p>
        </div>""", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""<div class="kpi-card"><div class="kpi-label">Sante Systeme</div>
            <div class="kpi-value">92%</div><div class="kpi-sub green">Excellent</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="kpi-card"><div class="kpi-label">RUL Predit</div>
            <div class="kpi-value">128</div><div class="kpi-sub blue">cycles — Moteur 1</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="kpi-card"><div class="kpi-label">Score Anomalie</div>
            <div class="kpi-value">18.6</div><div class="kpi-sub green">Normal</div></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown("""<div class="kpi-card"><div class="kpi-label">Alertes Actives</div>
            <div class="kpi-value">2</div><div class="kpi-sub orange">Avertissements</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns([2.2, 1])
        with col_left:
            st.markdown("### Tendance RUL")
            days    = list(range(1, 18))
            rul_sim = [200,185,172,160,148,137,128,118,108,98,87,75,62,50,38,25,12]
            fig, ax = plt.subplots(figsize=(9, 3.2))
            ax.fill_between(days, rul_sim, alpha=0.15, color='#2d7a4f')
            ax.plot(days, rul_sim, color='#2d7a4f', linewidth=2.5, marker='o', markersize=6)
            ax.axhline(y=30, color='#c0392b', linestyle='--', linewidth=1.5, label='Seuil critique')
            ax.set_xlabel('Jours', fontsize=11)
            ax.set_ylabel('RUL (cycles)', fontsize=11)
            ax.set_facecolor('#faf7f2')
            fig.patch.set_facecolor('white')
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

        with col_right:
            st.markdown("### Alertes Recentes")
            st.markdown("""
            <div class="alert-high"><b>Temperature Elevee</b><br><small style="color:#888;">Turbine 3 — il y a 2 min</small></div>
            <div class="alert-warn"><b>Vibration Anormale</b><br><small style="color:#888;">Compresseur 2 — il y a 15 min</small></div>
            <div class="alert-ok"><b>Systeme Normal</b><br><small style="color:#888;">Generateur 1 — il y a 1h</small></div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Performance Modeles")
            st.markdown("""
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <span style="font-weight:600;color:#1a4d2e;">Random Forest</span>
                    <span class="badge badge-green">ML</span>
                </div>
                <div style="display:flex;gap:8px;">
                    <div style="flex:1;text-align:center;background:#f5f0e8;border-radius:10px;padding:10px;">
                        <div style="font-size:1rem;font-weight:700;color:#1a4d2e;">35.85</div>
                        <div style="font-size:0.7rem;color:#888;">RMSE</div>
                    </div>
                    <div style="flex:1;text-align:center;background:#f5f0e8;border-radius:10px;padding:10px;">
                        <div style="font-size:1rem;font-weight:700;color:#1a4d2e;">25.34</div>
                        <div style="font-size:0.7rem;color:#888;">MAE</div>
                    </div>
                    <div style="flex:1;text-align:center;background:#f5f0e8;border-radius:10px;padding:10px;">
                        <div style="font-size:1rem;font-weight:700;color:#2d7a4f;">0.72</div>
                        <div style="font-size:0.7rem;color:#888;">R2</div>
                    </div>
                </div>
            </div>
            <div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <span style="font-weight:600;color:#1a4d2e;">LSTM</span>
                    <span class="badge badge-green">Deep Learning</span>
                </div>
                <div style="display:flex;gap:8px;">
                    <div style="flex:1;text-align:center;background:#f5f0e8;border-radius:10px;padding:10px;">
                        <div style="font-size:1rem;font-weight:700;color:#2d7a4f;">14.92</div>
                        <div style="font-size:0.7rem;color:#888;">RMSE</div>
                    </div>
                    <div style="flex:1;text-align:center;background:#f5f0e8;border-radius:10px;padding:10px;">
                        <div style="font-size:1rem;font-weight:700;color:#2d7a4f;">11.16</div>
                        <div style="font-size:0.7rem;color:#888;">MAE</div>
                    </div>
                    <div style="flex:1;text-align:center;background:#f5f0e8;border-radius:10px;padding:10px;">
                        <div style="font-size:1rem;font-weight:700;color:#2d7a4f;">0.8613</div>
                        <div style="font-size:0.7rem;color:#888;">R2</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif page == "Statistiques Avancees":
        st.markdown("""<div class="page-header">
            <h1>Analyse Statistique Avancee</h1>
            <p>Groupe OCP — Exploration approfondie des donnees NASA CMAPSS FD001</p>
        </div>""", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""<div class="kpi-card"><div class="kpi-label">Observations</div>
            <div class="kpi-value">20 631</div><div class="kpi-sub blue">lignes train</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="kpi-card"><div class="kpi-label">Moteurs</div>
            <div class="kpi-value">100</div><div class="kpi-sub green">unites</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="kpi-card"><div class="kpi-label">Capteurs Utiles</div>
            <div class="kpi-value">15</div><div class="kpi-sub green">sur 21</div></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown("""<div class="kpi-card"><div class="kpi-label">RUL Moyen</div>
            <div class="kpi-value">108</div><div class="kpi-sub blue">cycles</div></div>""", unsafe_allow_html=True)

        for img_file, title in [('rul_distribution.png','Distribution du RUL'),('heatmap.png','Heatmap de Correlation')]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### " + title)
            img_p = os.path.join('..', 'models', img_file)
            if os.path.exists(img_p):
                st.image(img_p, use_container_width=True)
            else:
                st.warning("Generez d abord les images dans 01_EDA.ipynb")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Distribution des Capteurs")
            img_d = os.path.join('..', 'models', 'distributions.png')
            if os.path.exists(img_d): st.image(img_d, use_container_width=True)
        with col2:
            st.markdown("### Detection des Outliers")
            img_b = os.path.join('..', 'models', 'boxplots.png')
            if os.path.exists(img_b): st.image(img_b, use_container_width=True)

    elif page == "Prediction RUL":
        st.markdown("""<div class="page-header">
            <h1>Prediction de la Duree de Vie Restante</h1>
            <p>Groupe OCP — Entrez les valeurs des capteurs pour estimer le RUL</p>
        </div>""", unsafe_allow_html=True)

        if rf_model is None or scaler is None:
            st.error("Modele non disponible. Executez d abord le notebook 02_MODEL.ipynb")
        else:
            useless  = ['s1', 's5', 's10', 's16', 's18', 's19']
            useful   = [s for s in ['s' + str(i) for i in range(1, 22)] if s not in useless]
            features = ['cycle'] + useful
            col1, col2 = st.columns(2)
            inputs = {}
            inputs['cycle'] = col1.number_input('Cycle', min_value=1, max_value=500, value=100, key="inp_cycle")
            for i, feat in enumerate(useful):
                if i % 2 == 0:
                    inputs[feat] = col1.number_input(feat, value=float(50), format="%.4f", key="inp_" + feat)
                else:
                    inputs[feat] = col2.number_input(feat, value=float(50), format="%.4f", key="inp_" + feat)
            st.markdown("---")
            if st.button("Predire le RUL", key="btn_predict"):
                try:
                    X_input  = pd.DataFrame([inputs])[features]
                    X_scaled = scaler.transform(X_input)
                    rul_pred = rf_model.predict(X_scaled)[0]
                    if rul_pred > 100:
                        st.success("RUL estime : " + str(int(rul_pred)) + " cycles — Equipement en bon etat")
                        color = "#2d7a4f"; badge = "BON ETAT"; badge_class = "badge-green"
                    elif rul_pred > 30:
                        st.warning("RUL estime : " + str(int(rul_pred)) + " cycles — Maintenance recommandee")
                        color = "#c8860a"; badge = "MAINTENANCE"; badge_class = "badge-orange"
                    else:
                        st.error("RUL estime : " + str(int(rul_pred)) + " cycles — PANNE IMMINENTE")
                        color = "#c0392b"; badge = "CRITIQUE"; badge_class = "badge-red"
                    pct = min(rul_pred / 361 * 100, 100)
                    st.markdown(
                        '<div style="margin-top:16px;background:white;border-radius:16px;padding:20px;">'
                        '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                        '<span style="font-weight:600;color:#1a4d2e;">Etat de l equipement</span>'
                        '<span class="badge ' + badge_class + '">' + badge + '</span></div>'
                        '<div class="progress-bar" style="height:14px;">'
                        '<div class="progress-fill" style="width:' + str(int(pct)) + '%;background:' + color + ';"></div></div>'
                        '<div style="font-size:0.82rem;color:#888;margin-top:6px;">RUL: ' + str(int(rul_pred)) + ' / 361 cycles max</div>'
                        '</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error("Erreur : " + str(e))

    elif page == "SHAP Explainability":
        st.markdown("""<div class="page-header">
            <h1>Explainability — SHAP</h1>
            <p>Groupe OCP — Comprendre les decisions du modele</p>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Importance globale des features")
            img1 = os.path.join('..', 'models', 'shap_importance.png')
            if os.path.exists(img1): st.image(img1, use_container_width=True)
        with col2:
            st.markdown("### Impact des features sur le RUL")
            img2 = os.path.join('..', 'models', 'shap_beeswarm.png')
            if os.path.exists(img2): st.image(img2, use_container_width=True)

        st.markdown("---")
        st.markdown("### Interpretation des Resultats")
        st.markdown("""
        | Feature | Importance SHAP | Interpretation |
        |---------|----------------|----------------|
        | **cycle** | ~38 | Variable la plus predictive |
        | **s11**   | ~11 | Capteur critique |
        | **s9, s12** | ~4-5 | Capteurs secondaires |
        | **s4, s14** | ~3-4 | Contribution moderee |
        """)

    elif page == "LSTM Deep Learning":
        st.markdown("""<div class="page-header">
            <h1>LSTM — Deep Learning</h1>
            <p>Groupe OCP — Prediction temporelle de la Duree de Vie Restante</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;">
            <div style="flex:1;min-width:130px;background:white;border-radius:14px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(45,122,79,0.08);border-top:3px solid #2d7a4f;">
                <div style="font-size:0.8rem;color:#888;">Sequence</div><div style="font-size:1.6rem;font-weight:700;color:#1a4d2e;">30</div>
                <div style="font-size:0.75rem;color:#2d7a4f;">cycles</div>
            </div>
            <div style="flex:1;min-width:130px;background:white;border-radius:14px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(45,122,79,0.08);border-top:3px solid #2d7a4f;">
                <div style="font-size:0.8rem;color:#888;">LSTM Layer 1</div><div style="font-size:1.6rem;font-weight:700;color:#1a4d2e;">128</div>
                <div style="font-size:0.75rem;color:#2d7a4f;">neurones</div>
            </div>
            <div style="flex:1;min-width:130px;background:white;border-radius:14px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(45,122,79,0.08);border-top:3px solid #2d7a4f;">
                <div style="font-size:0.8rem;color:#888;">LSTM Layer 2</div><div style="font-size:1.6rem;font-weight:700;color:#1a4d2e;">64</div>
                <div style="font-size:0.75rem;color:#2d7a4f;">neurones</div>
            </div>
            <div style="flex:1;min-width:130px;background:white;border-radius:14px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(45,122,79,0.08);border-top:3px solid #c8860a;">
                <div style="font-size:0.8rem;color:#888;">Dropout</div><div style="font-size:1.6rem;font-weight:700;color:#c8860a;">20%</div>
                <div style="font-size:0.75rem;color:#c8860a;">regularisation</div>
            </div>
            <div style="flex:1;min-width:130px;background:white;border-radius:14px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(45,122,79,0.08);border-top:3px solid #c8860a;">
                <div style="font-size:0.8rem;color:#888;">R2 Score</div><div style="font-size:1.6rem;font-weight:700;color:#2d7a4f;">0.8613</div>
                <div style="font-size:0.75rem;color:#2d7a4f;">LSTM</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        img_curves = os.path.join('..', 'models', 'lstm_training_curves.png')
        img_pred   = os.path.join('..', 'models', 'lstm_prediction.png')
        if os.path.exists(img_curves):
            st.markdown("### Courbes d Apprentissage LSTM")
            st.image(img_curves, use_container_width=True)
        if os.path.exists(img_pred):
            st.markdown("### Prediction vs Reel — LSTM")
            st.image(img_pred, use_container_width=True)

    elif page == "Chatbot":
        st.markdown("""<div class="page-header">
            <h1>Chatbot PredictIQ — Maintenance</h1>
            <p>Groupe OCP — Posez vos questions sur la maintenance predictive</p>
        </div>""", unsafe_allow_html=True)

        def get_response_m(question):
            q = question.lower().strip()
            if any(w in q for w in ["rul", "duree", "remaining", "vie", "cycles"]):
                return ("RUL — Remaining Useful Life<br><br>"
                        "RF : R2=0.72 — RMSE=35.85<br>"
                        "LSTM : R2=0.8613 — RMSE=14.92<br><br>"
                        "RUL sup 100 : Bon etat<br>"
                        "RUL 30-100 : Maintenance<br>"
                        "RUL inf 30 : Critique")
            elif any(w in q for w in ["lstm", "deep", "neurone"]):
                return ("LSTM — R2=0.8613 | RMSE=14.92<br>"
                        "Sequence 30 cycles<br>"
                        "128 + 64 neurones<br>"
                        "Dropout 20%")
            elif any(w in q for w in ["shap", "feature", "capteur"]):
                return ("Top SHAP features :<br>"
                        "1. cycle ~38<br>"
                        "2. s11 ~11<br>"
                        "3. s9 ~5<br>"
                        "4. s12 ~4")
            elif any(w in q for w in ["bonjour", "salut", "aide", "salam"]):
                return ("Bonjour — Chatbot Maintenance PredictIQ<br><br>"
                        "Je reponds sur : RUL, LSTM, SHAP, Dataset")
            else:
                return ("Question non comprise.<br>"
                        "Essayez : RUL, LSTM, SHAP, capteurs")

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown('<div style="display:flex;justify-content:flex-end;margin:8px 0;"><div style="background:#1a4d2e;color:white;border-radius:18px 18px 4px 18px;padding:12px 18px;max-width:70%;font-size:0.9rem;">' + msg["content"] + '</div><div style="margin-left:8px;font-size:1.5rem;line-height:2;">U</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="display:flex;justify-content:flex-start;margin:8px 0;"><div style="margin-right:8px;font-size:1.5rem;line-height:2;">B</div><div style="background:white;border:2px solid #e8e0d0;border-radius:18px 18px 18px 4px;padding:12px 18px;max-width:75%;font-size:0.9rem;">' + msg["content"] + '</div></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("q", placeholder="Posez votre question...", label_visibility="collapsed", key="chat_m")
        with col2:
            send = st.button("Envoyer", key="send_m")

        if send and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": get_response_m(user_input)})
            st.rerun()

        if st.button("Effacer", key="clear_m"):
            st.session_state.messages = [{"role": "assistant", "content": "Bonjour — Je suis le chatbot PredictIQ. Comment puis-je vous aider ?"}]
            st.rerun()

# ══════════════════════════════════════════════════════
# MODULE OPTIMISATION ENERGETIQUE
# ══════════════════════════════════════════════════════
elif st.session_state.module == "energie":

    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:10px 0 24px 0;">'
            + logo_html_sm +
            '<div style="font-size:1.3rem;font-weight:800;color:white;letter-spacing:1px;">PredictIQ</div>'
            '<div style="font-size:0.7rem;color:#a8d5b5;margin-top:2px;font-style:italic;">Optimisation Energetique</div>'
            '</div>', unsafe_allow_html=True)

        st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:0 0 16px 0;'>", unsafe_allow_html=True)

        st.markdown("""
        <div style='padding:12px 16px;background:rgba(255,255,255,0.10);border-radius:12px;margin-bottom:20px;'>
            <div style='font-size:0.75rem;color:#a8d5b5;'>Groupe OCP</div>
            <div style='font-size:0.95rem;font-weight:600;color:white;margin-top:2px;'>OCP Laayoune</div>
            <div style='font-size:0.75rem;color:#a8d5b5;margin-top:4px;'>Mai 2026</div>
        </div>
        """, unsafe_allow_html=True)

        page_e = st.radio("Navigation", [
            "Dashboard Energie",
            "Detection Surconsommation",
            "Prediction Consommation",
            "Recommandations",
            "Chatbot Energie"
        ], key="nav_energie")

        st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:16px 0;'>", unsafe_allow_html=True)

        if st.button("Accueil", key="btn_accueil_e"):
            st.session_state.module = None
            st.rerun()

        st.markdown(
            '<div style="padding:12px;background:rgba(255,255,255,0.12);border-radius:12px;text-align:center;margin-top:12px;">'
            '<div style="color:#a8d5b5;font-weight:600;font-size:0.85rem;">R2 = 0.9991</div>'
            '<div style="color:#c8dfc8;font-size:0.75rem;margin-top:4px;">Steel Industry Dataset</div>'
            '</div>', unsafe_allow_html=True)

    # ── Dashboard Energie ──
    if page_e == "Dashboard Energie":
        st.markdown("""<div class="page-header-energy">
            <h1>Dashboard — Optimisation Energetique</h1>
            <p>Groupe OCP — Analyse et optimisation de la consommation electrique industrielle</p>
        </div>""", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="kpi-card-energy"><div class="kpi-label">Economies Potentielles</div><div class="kpi-value-energy">' + str(round(reco['economies_mad'])) + '</div><div class="kpi-sub orange">MAD / an</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="kpi-card-energy"><div class="kpi-label">Economies kWh</div><div class="kpi-value-energy">' + str(round(reco['economies_kwh'])) + '</div><div class="kpi-sub orange">kWh detectes</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="kpi-card-energy"><div class="kpi-label">Anomalies Detectees</div><div class="kpi-value-energy">' + str(reco['n_anomalies']) + '</div><div class="kpi-sub orange">surconsommations</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="kpi-card-energy"><div class="kpi-label">R2 Score Modele</div><div class="kpi-value-energy">0.9991</div><div class="kpi-sub green">Excellent</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        img_eda = os.path.join('..', 'models', 'energie_eda.png')
        if os.path.exists(img_eda):
            st.markdown("### Analyse Consommation Energetique")
            st.image(img_eda, use_container_width=True)
        else:
            st.info("Executez d abord le notebook 05_ENERGIE.ipynb")

    # ── Detection Surconsommation ──
    elif page_e == "Detection Surconsommation":
        st.markdown("""<div class="page-header-energy">
            <h1>Detection des Surconsommations</h1>
            <p>Groupe OCP — Isolation Forest — Detection automatique des anomalies energetiques</p>
        </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="kpi-card-energy"><div class="kpi-label">Conso Normale Moyenne</div><div class="kpi-value-energy">' + str(reco['conso_normale']) + '</div><div class="kpi-sub green">kWh</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="kpi-card-energy"><div class="kpi-label">Conso Anomalie Moyenne</div><div class="kpi-value-energy">' + str(reco['conso_anomalie']) + '</div><div class="kpi-sub orange">kWh</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="kpi-card-energy"><div class="kpi-label">Anomalies Detectees</div><div class="kpi-value-energy">' + str(reco['n_anomalies']) + '</div><div class="kpi-sub orange">evenements</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        img_an = os.path.join('..', 'models', 'energie_anomalies.png')
        if os.path.exists(img_an):
            st.image(img_an, use_container_width=True)
        else:
            st.info("Executez d abord le notebook 05_ENERGIE.ipynb")

        st.markdown("""
        <div style="background:linear-gradient(135deg,#e67e22,#f39c12);border-radius:16px;padding:20px;margin-top:16px;">
            <div style="font-size:1rem;font-weight:600;margin-bottom:8px;color:white;">Methode — Isolation Forest</div>
            <div style="font-size:0.9rem;color:#fdebd0;line-height:1.6;">
                L algorithme Isolation Forest detecte les <b style="color:white;">surconsommations anormales</b>
                en isolant les points aberrants dans l espace multidimensionnel des capteurs energetiques.
                Contamination fixee a <b style="color:white;">5%</b> du dataset.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Prediction Consommation ──
    elif page_e == "Prediction Consommation":
        st.markdown("""<div class="page-header-energy">
            <h1>Prediction de Consommation</h1>
            <p>Groupe OCP — Random Forest — R2 = 0.9991</p>
        </div>""", unsafe_allow_html=True)

        img_pred = os.path.join('..', 'models', 'energie_prediction.png')
        if os.path.exists(img_pred):
            st.image(img_pred, use_container_width=True)
        else:
            st.info("Executez d abord le notebook 05_ENERGIE.ipynb")

        st.markdown("### Performance du Modele")
        st.markdown("""
        <div style="background:white;border-radius:16px;padding:20px;box-shadow:0 4px 20px rgba(230,126,34,0.08);">
            <table style="width:100%;border-collapse:collapse;">
                <thead><tr style="background:#fef9f0;">
                    <th style="padding:12px;text-align:left;color:#e67e22;">Metrique</th>
                    <th style="padding:12px;text-align:center;color:#e67e22;">Valeur</th>
                    <th style="padding:12px;text-align:center;color:#e67e22;">Interpretation</th>
                </tr></thead>
                <tbody>
                    <tr style="border-bottom:1px solid #fef0e0;">
                        <td style="padding:12px;font-weight:600;">RMSE</td>
                        <td style="padding:12px;text-align:center;font-weight:700;">1.0055 kWh</td>
                        <td style="padding:12px;text-align:center;color:#2d7a4f;">Tres faible erreur</td>
                    </tr>
                    <tr style="border-bottom:1px solid #fef0e0;">
                        <td style="padding:12px;font-weight:600;">MAE</td>
                        <td style="padding:12px;text-align:center;font-weight:700;">0.3529 kWh</td>
                        <td style="padding:12px;text-align:center;color:#2d7a4f;">Excellent</td>
                    </tr>
                    <tr>
                        <td style="padding:12px;font-weight:600;">R2</td>
                        <td style="padding:12px;text-align:center;font-weight:700;color:#e67e22;">0.9991</td>
                        <td style="padding:12px;text-align:center;color:#2d7a4f;">Quasi parfait</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ── Recommandations ──
    elif page_e == "Recommandations":
        st.markdown("""<div class="page-header-energy">
            <h1>Recommandations d Optimisation</h1>
            <p>Groupe OCP — Actions concretes pour reduire la consommation energetique</p>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background:white;border-radius:16px;padding:24px;box-shadow:0 4px 20px rgba(230,126,34,0.08);border-top:4px solid #e67e22;margin-bottom:16px;">
                <div style="font-size:1.1rem;font-weight:700;color:#e67e22;margin-bottom:12px;">Gestion des Heures</div>
                <div style="font-size:0.9rem;color:#555;line-height:1.8;">
                    <b>Heure de pointe :</b> """ + str(reco['heure_pointe']) + """h<br>
                    Reduire la charge electrique durant cette heure<br><br>
                    <b>Heure creuse :</b> """ + str(reco['heure_creuse']) + """h<br>
                    Programmer les taches lourdes a cette heure
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background:white;border-radius:16px;padding:24px;box-shadow:0 4px 20px rgba(230,126,34,0.08);border-top:4px solid #c0392b;">
                <div style="font-size:1.1rem;font-weight:700;color:#c0392b;margin-bottom:12px;">Anomalies a Corriger</div>
                <div style="font-size:0.9rem;color:#555;line-height:1.8;">
                    <b>""" + str(reco['n_anomalies']) + """ evenements</b> de surconsommation detectes<br><br>
                    Actions recommandees :<br>
                    Verifier les equipements en surcharge<br>
                    Inspecter les circuits electriques<br>
                    Calibrer les capteurs de puissance
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(
                '<div style="background:linear-gradient(135deg,#e67e22,#f39c12);border-radius:16px;padding:24px;color:white;margin-bottom:16px;">'
                '<div style="font-size:1.1rem;font-weight:700;margin-bottom:16px;">Economies Potentielles</div>'
                '<div style="display:flex;justify-content:space-between;margin-bottom:12px;">'
                '<span style="color:#fdebd0;">Economies kWh</span>'
                '<span style="font-weight:700;font-size:1.1rem;">' + str(round(reco['economies_kwh'])) + ' kWh</span>'
                '</div>'
                '<div style="display:flex;justify-content:space-between;margin-bottom:12px;">'
                '<span style="color:#fdebd0;">Prix kWh</span>'
                '<span style="font-weight:700;">' + str(reco['prix_kwh_mad']) + ' MAD</span>'
                '</div>'
                '<div style="border-top:1px solid rgba(255,255,255,0.3);padding-top:12px;margin-top:12px;">'
                '<div style="display:flex;justify-content:space-between;">'
                '<span style="color:#fdebd0;font-size:1rem;">Total Economies</span>'
                '<span style="font-weight:900;font-size:1.8rem;">' + str(round(reco['economies_mad'])) + ' MAD</span>'
                '</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown("""
            <div style="background:white;border-radius:16px;padding:24px;box-shadow:0 4px 20px rgba(230,126,34,0.08);border-top:4px solid #2d7a4f;">
                <div style="font-size:1.1rem;font-weight:700;color:#2d7a4f;margin-bottom:12px;">Bonnes Pratiques</div>
                <div style="font-size:0.9rem;color:#555;line-height:1.8;">
                    Surveillance continue des capteurs<br>
                    Maintenance preventive reguliere<br>
                    Formation du personnel<br>
                    Optimisation des cycles de production<br>
                    Audit energetique periodique
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Chatbot Energie ──
    elif page_e == "Chatbot Energie":
        st.markdown("""<div class="page-header-energy">
            <h1>Chatbot — Optimisation Energetique</h1>
            <p>Groupe OCP — Posez vos questions sur l optimisation energetique</p>
        </div>""", unsafe_allow_html=True)

        def get_response_e(question):
            q = question.lower().strip()
            if any(w in q for w in ["economie", "mad", "dirham", "kwh", "argent"]):
                return ("Economies Potentielles<br><br>"
                        "kWh : " + str(round(reco['economies_kwh'])) + "<br>"
                        "MAD : " + str(round(reco['economies_mad'])) + "<br>"
                        "Prix kWh : " + str(reco['prix_kwh_mad']) + " MAD<br><br>"
                        "Basees sur " + str(reco['n_anomalies']) + " anomalies detectees")
            elif any(w in q for w in ["anomalie", "surconso", "detection", "isolation"]):
                return ("Detection Surconsommation<br><br>"
                        "Methode : Isolation Forest<br>"
                        "Anomalies : " + str(reco['n_anomalies']) + " evenements<br>"
                        "Contamination : 5%<br><br>"
                        "Conso normale : " + str(reco['conso_normale']) + " kWh<br>"
                        "Conso anomalie : " + str(reco['conso_anomalie']) + " kWh")
            elif any(w in q for w in ["heure", "pointe", "creuse", "pic"]):
                return ("Gestion des Heures<br><br>"
                        "Heure de pointe : " + str(reco['heure_pointe']) + "h<br>"
                        "Reduire la charge a cette heure<br><br>"
                        "Heure creuse : " + str(reco['heure_creuse']) + "h<br>"
                        "Programmer les taches lourdes")
            elif any(w in q for w in ["modele", "r2", "rmse", "precision"]):
                return ("Modele RF Energie<br><br>"
                        "R2   : 0.9991<br>"
                        "RMSE : 1.0055 kWh<br>"
                        "MAE  : 0.3529 kWh<br><br>"
                        "Precision quasi parfaite !")
            elif any(w in q for w in ["recommandation", "conseil", "action"]):
                return ("Recommandations<br><br>"
                        "1. Reduire charge a " + str(reco['heure_pointe']) + "h<br>"
                        "2. Taches lourdes a " + str(reco['heure_creuse']) + "h<br>"
                        "3. Corriger " + str(reco['n_anomalies']) + " anomalies<br>"
                        "4. Economies : " + str(round(reco['economies_mad'])) + " MAD")
            elif any(w in q for w in ["bonjour", "salut", "aide", "salam"]):
                return ("Bonjour — Chatbot Energie PredictIQ<br><br>"
                        "Je reponds sur :<br>"
                        "Economies — Anomalies — Heures<br>"
                        "Modele — Recommandations")
            else:
                return ("Question non comprise.<br>"
                        "Essayez : economies, anomalies,<br>"
                        "heures, modele, recommandations")

        for msg in st.session_state.messages_e:
            if msg["role"] == "user":
                st.markdown('<div style="display:flex;justify-content:flex-end;margin:8px 0;"><div style="background:#e67e22;color:white;border-radius:18px 18px 4px 18px;padding:12px 18px;max-width:70%;font-size:0.9rem;">' + msg["content"] + '</div><div style="margin-left:8px;font-size:1.5rem;line-height:2;">U</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="display:flex;justify-content:flex-start;margin:8px 0;"><div style="margin-right:8px;font-size:1.5rem;line-height:2;">B</div><div style="background:white;border:2px solid #fdebd0;border-radius:18px 18px 18px 4px;padding:12px 18px;max-width:75%;font-size:0.9rem;">' + msg["content"] + '</div></div>', unsafe_allow_html=True)

        col1, col2 = st.columns([5, 1])
        with col1:
            user_input_e = st.text_input("q", placeholder="Posez votre question...", label_visibility="collapsed", key="chat_e")
        with col2:
            send_e = st.button("Envoyer", key="send_e")

        st.markdown("**Questions rapides :**")
        qc1, qc2, qc3, qc4 = st.columns(4)
        quick_e = None
        with qc1:
            if st.button("Economies", key="qe1"): quick_e = "economies en mad"
        with qc2:
            if st.button("Anomalies", key="qe2"): quick_e = "anomalies detectees"
        with qc3:
            if st.button("Heures", key="qe3"): quick_e = "heures pointe creuse"
        with qc4:
            if st.button("Recommandations", key="qe4"): quick_e = "recommandations"

        final_e = quick_e if quick_e else (user_input_e if send_e and user_input_e.strip() else None)
        if final_e:
            st.session_state.messages_e.append({"role": "user", "content": final_e})
            st.session_state.messages_e.append({"role": "assistant", "content": get_response_e(final_e)})
            st.rerun()

        if st.button("Effacer", key="clear_e"):
            st.session_state.messages_e = [{"role": "assistant", "content": "Bonjour — Je suis le chatbot Energie PredictIQ. Comment puis-je vous aider ?"}]
            st.rerun()