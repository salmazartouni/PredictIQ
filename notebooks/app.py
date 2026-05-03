import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import base64

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
.kpi-value { font-size: 2.1rem; font-weight: 700; color: #1a4d2e; }
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
.page-header h1 { color: white !important; font-size: 1.8rem !important; margin: 0 !important; font-weight: 700 !important; }
.page-header p  { color: #a8d5b5 !important; margin: 6px 0 0 0 !important; font-size: 0.95rem !important; }
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
[data-testid="stMetric"] { background: white; border-radius: 16px; padding: 18px; box-shadow: 0 4px 20px rgba(45,122,79,0.08); }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

MODEL_PATH  = os.path.join('..', 'models', 'rf_model.pkl')
SCALER_PATH = os.path.join('..', 'models', 'scaler.pkl')

@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
        return None, None
    except Exception:
        return None, None

rf_model, scaler = load_model()

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

if "page_garde_shown" not in st.session_state:
    st.session_state.page_garde_shown = False
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour — Je suis le chatbot PredictIQ. Comment puis-je vous aider ?"}
    ]

# ══════════════════════════════════════════════════════
# PAGE DE GARDE
# ══════════════════════════════════════════════════════
if not st.session_state.page_garde_shown:

    st.markdown(
        '<div style="min-height:90vh;display:flex;flex-direction:column;justify-content:center;align-items:center;padding:40px 20px;">'
        '<div style="text-align:center;margin-bottom:40px;">'
        '<div style="display:inline-block;background:white;border-radius:20px;padding:16px 40px;box-shadow:0 4px 24px rgba(45,122,79,0.12);border-bottom:3px solid #2d7a4f;">'
        '<div style="font-size:0.8rem;color:#2d7a4f;font-weight:700;letter-spacing:3px;text-transform:uppercase;">Groupe OCP — OCP Phosboucraa</div>'
        '</div></div>'
        '<div style="background:linear-gradient(135deg,#1a4d2e 0%,#2d7a4f 60%,#3d9b6a 100%);border-radius:32px;padding:60px 80px;max-width:800px;width:100%;box-shadow:0 30px 80px rgba(26,77,46,0.35);text-align:center;">'
        '<div style="font-size:3.5rem;margin-bottom:20px;">OCP</div>'
        '<div style="font-size:3.8rem;font-weight:900;color:white;letter-spacing:4px;margin-bottom:8px;">PredictIQ</div>'
        '<div style="font-size:1rem;color:#c8f0d8;font-weight:500;letter-spacing:1px;margin-bottom:30px;">Intelligent Predictive Maintenance Platform</div>'
        '<div style="width:80px;height:3px;background:rgba(255,255,255,0.4);border-radius:2px;margin:0 auto 30px auto;"></div>'
        '<div style="font-size:0.95rem;color:#a8d5b5;line-height:1.9;margin-bottom:40px;">'
        'Analyse Statistique Avancee &nbsp;•&nbsp; Machine Learning &nbsp;•&nbsp; Deep Learning LSTM<br>'
        'Explainable AI (SHAP) &nbsp;•&nbsp; Chatbot Interactif'
        '</div>'
        '<div style="display:flex;justify-content:center;gap:40px;margin-bottom:40px;flex-wrap:wrap;">'
        '<div style="text-align:center;"><div style="font-size:1.8rem;font-weight:800;color:white;">0.8613</div><div style="font-size:0.75rem;color:#a8d5b5;margin-top:2px;">R2 Score LSTM</div></div>'
        '<div style="width:1px;background:rgba(255,255,255,0.2);"></div>'
        '<div style="text-align:center;"><div style="font-size:1.8rem;font-weight:800;color:white;">14.92</div><div style="font-size:0.75rem;color:#a8d5b5;margin-top:2px;">RMSE (cycles)</div></div>'
        '<div style="width:1px;background:rgba(255,255,255,0.2);"></div>'
        '<div style="text-align:center;"><div style="font-size:1.8rem;font-weight:800;color:white;">100</div><div style="font-size:0.75rem;color:#a8d5b5;margin-top:2px;">Moteurs analyses</div></div>'
        '<div style="width:1px;background:rgba(255,255,255,0.2);"></div>'
        '<div style="text-align:center;"><div style="font-size:1.8rem;font-weight:800;color:white;">20K+</div><div style="font-size:0.75rem;color:#a8d5b5;margin-top:2px;">Observations</div></div>'
        '</div>'
        '<div style="background:rgba(255,255,255,0.1);border-radius:12px;padding:12px 24px;display:inline-block;">'
        '<div style="color:#c8f0d8;font-size:0.85rem;font-weight:500;">NASA CMAPSS FD001 &nbsp;|&nbsp; Random Forest &nbsp;+&nbsp; LSTM Deep Learning</div>'
        '</div>'
        '</div>'
        '<div style="margin-top:28px;text-align:center;">'
        '<div style="font-size:0.9rem;color:#2d7a4f;font-style:italic;font-weight:500;">Anticipate. Explain. Prevent.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Acceder a la Plateforme", key="btn_acceder"):
            st.session_state.page_garde_shown = True
            st.rerun()

# ══════════════════════════════════════════════════════
# APPLICATION PRINCIPALE
# ══════════════════════════════════════════════════════
else:

    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:10px 0 24px 0;">'
            + logo_html_sm +
            '<div style="font-size:1.5rem;font-weight:800;color:white;letter-spacing:1px;">PredictIQ</div>'
            '<div style="font-size:0.75rem;color:#a8d5b5;margin-top:4px;font-style:italic;">Anticipate. Explain. Prevent.</div>'
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
        ], key="nav_radio")

        st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:16px 0;'>", unsafe_allow_html=True)

        statut = "Modele charge" if rf_model is not None else "Modele non charge"
        couleur = "#a8d5b5" if rf_model is not None else "#c0392b"
        st.markdown(
            '<div style="padding:12px;background:rgba(255,255,255,0.12);border-radius:12px;text-align:center;">'
            '<div style="color:' + couleur + ';font-weight:600;font-size:0.85rem;">' + statut + '</div>'
            '<div style="color:#c8dfc8;font-size:0.75rem;margin-top:4px;">NASA CMAPSS FD001</div>'
            '</div>', unsafe_allow_html=True)

    if page == "Dashboard":
        col_h, col_btn = st.columns([8, 1])
        with col_h:
            st.markdown("""<div class="page-header">
                <h1>Tableau de Bord — Maintenance Predictive</h1>
                <p>Groupe OCP — Surveillance intelligente des equipements — NASA CMAPSS FD001</p>
            </div>""", unsafe_allow_html=True)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Accueil", key="btn_home1"):
                st.session_state.page_garde_shown = False
                st.rerun()

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
            ax.grid(True, alpha=0.2, color='#d4c9b0')
            ax.legend(fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)
            img_path = os.path.join('..', 'models', 'correlation_RUL.png')
            if os.path.exists(img_path):
                st.markdown("### Correlation Capteurs / RUL")
                st.image(img_path, use_container_width=True)

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
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Etat du Systeme")
            st.markdown("""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:0.85rem;color:#555;">En ligne</span>
                    <span style="font-size:0.85rem;font-weight:600;color:#2d7a4f;">22</span>
                </div>
                <div class="progress-bar"><div class="progress-fill" style="width:92%;"></div></div>
            </div>
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:0.85rem;color:#555;">Avertissement</span>
                    <span style="font-size:0.85rem;font-weight:600;color:#c8860a;">1</span>
                </div>
                <div class="progress-bar"><div class="progress-fill" style="width:4%;background:#c8860a;"></div></div>
            </div>
            <div>
                <div style="display:flex;justify-content:space-between;">
                    <span style="font-size:0.85rem;color:#555;">Critique</span>
                    <span style="font-size:0.85rem;font-weight:600;color:#c0392b;">0</span>
                </div>
                <div class="progress-bar"><div class="progress-fill" style="width:0%;"></div></div>
            </div>
            """, unsafe_allow_html=True)

    elif page == "Statistiques Avancees":
        col_h, col_btn = st.columns([8, 1])
        with col_h:
            st.markdown("""<div class="page-header">
                <h1>Analyse Statistique Avancee</h1>
                <p>Groupe OCP — Exploration approfondie des donnees NASA CMAPSS FD001</p>
            </div>""", unsafe_allow_html=True)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Accueil", key="btn_home2"):
                st.session_state.page_garde_shown = False
                st.rerun()

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

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Statistiques Descriptives")
        st.markdown("""
        <div style="display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;">
            <div style="flex:1;min-width:130px;background:white;border-radius:14px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(45,122,79,0.08);border-top:3px solid #2d7a4f;">
                <div style="font-size:0.8rem;color:#888;">RUL Minimum</div>
                <div style="font-size:1.6rem;font-weight:700;color:#1a4d2e;">0</div>
                <div style="font-size:0.75rem;color:#2d7a4f;">cycles</div>
            </div>
            <div style="flex:1;min-width:130px;background:white;border-radius:14px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(45,122,79,0.08);border-top:3px solid #2d7a4f;">
                <div style="font-size:0.8rem;color:#888;">RUL Maximum</div>
                <div style="font-size:1.6rem;font-weight:700;color:#1a4d2e;">361</div>
                <div style="font-size:0.75rem;color:#2d7a4f;">cycles</div>
            </div>
            <div style="flex:1;min-width:130px;background:white;border-radius:14px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(45,122,79,0.08);border-top:3px solid #2d7a4f;">
                <div style="font-size:0.8rem;color:#888;">RUL Moyenne</div>
                <div style="font-size:1.6rem;font-weight:700;color:#1a4d2e;">108.8</div>
                <div style="font-size:0.75rem;color:#2d7a4f;">cycles</div>
            </div>
            <div style="flex:1;min-width:130px;background:white;border-radius:14px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(45,122,79,0.08);border-top:3px solid #2d7a4f;">
                <div style="font-size:0.8rem;color:#888;">Ecart-type</div>
                <div style="font-size:1.6rem;font-weight:700;color:#1a4d2e;">68.6</div>
                <div style="font-size:0.75rem;color:#2d7a4f;">cycles</div>
            </div>
            <div style="flex:1;min-width:130px;background:white;border-radius:14px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(45,122,79,0.08);border-top:3px solid #c0392b;">
                <div style="font-size:0.8rem;color:#888;">Capteurs supprimes</div>
                <div style="font-size:1.6rem;font-weight:700;color:#c0392b;">6</div>
                <div style="font-size:0.75rem;color:#c0392b;">variance = 0</div>
            </div>
            <div style="flex:1;min-width:130px;background:white;border-radius:14px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(45,122,79,0.08);border-top:3px solid #c8860a;">
                <div style="font-size:0.8rem;color:#888;">Valeurs manquantes</div>
                <div style="font-size:1.6rem;font-weight:700;color:#c8860a;">0</div>
                <div style="font-size:0.75rem;color:#c8860a;">donnees propres</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for img_file, title in [('rul_distribution.png','Distribution du RUL'),('heatmap.png','Heatmap de Correlation')]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### " + title)
            img_p = os.path.join('..', 'models', img_file)
            if os.path.exists(img_p):
                st.image(img_p, use_container_width=True)
            else:
                st.warning("Generez d abord les images dans 01_EDA.ipynb")

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Distribution des Capteurs")
            img_d = os.path.join('..', 'models', 'distributions.png')
            if os.path.exists(img_d): st.image(img_d, use_container_width=True)
            else: st.warning("Generez d abord les images dans 01_EDA.ipynb")
        with col2:
            st.markdown("### Detection des Outliers")
            img_b = os.path.join('..', 'models', 'boxplots.png')
            if os.path.exists(img_b): st.image(img_b, use_container_width=True)
            else: st.warning("Generez d abord les images dans 01_EDA.ipynb")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Capteurs Supprimes — Variance = 0")
        st.markdown("""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;">
            <div style="background:#ffebee;border:2px solid #c0392b;border-radius:10px;padding:10px 20px;text-align:center;"><div style="font-weight:700;color:#c0392b;font-size:1.1rem;">s1</div><div style="font-size:0.75rem;color:#888;">variance = 0</div></div>
            <div style="background:#ffebee;border:2px solid #c0392b;border-radius:10px;padding:10px 20px;text-align:center;"><div style="font-weight:700;color:#c0392b;font-size:1.1rem;">s5</div><div style="font-size:0.75rem;color:#888;">variance = 0</div></div>
            <div style="background:#ffebee;border:2px solid #c0392b;border-radius:10px;padding:10px 20px;text-align:center;"><div style="font-weight:700;color:#c0392b;font-size:1.1rem;">s10</div><div style="font-size:0.75rem;color:#888;">variance = 0</div></div>
            <div style="background:#ffebee;border:2px solid #c0392b;border-radius:10px;padding:10px 20px;text-align:center;"><div style="font-weight:700;color:#c0392b;font-size:1.1rem;">s16</div><div style="font-size:0.75rem;color:#888;">variance = 0</div></div>
            <div style="background:#ffebee;border:2px solid #c0392b;border-radius:10px;padding:10px 20px;text-align:center;"><div style="font-weight:700;color:#c0392b;font-size:1.1rem;">s18</div><div style="font-size:0.75rem;color:#888;">variance = 0</div></div>
            <div style="background:#ffebee;border:2px solid #c0392b;border-radius:10px;padding:10px 20px;text-align:center;"><div style="font-weight:700;color:#c0392b;font-size:1.1rem;">s19</div><div style="font-size:0.75rem;color:#888;">variance = 0</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a4d2e,#2d7a4f);border-radius:16px;padding:20px;margin-top:16px;">
            <div style="font-size:1rem;font-weight:600;margin-bottom:8px;color:white;">Conclusion Statistique</div>
            <div style="font-size:0.9rem;color:#a8d5b5;line-height:1.6;">
                L analyse statistique avancee a permis d identifier <b style="color:white;">6 capteurs inutiles</b>,
                de detecter les <b style="color:white;">correlations significatives</b> entre capteurs et RUL,
                et de confirmer l absence de valeurs manquantes.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif page == "Prediction RUL":
        col_h, col_btn = st.columns([8, 1])
        with col_h:
            st.markdown("""<div class="page-header">
                <h1>Prediction de la Duree de Vie Restante</h1>
                <p>Groupe OCP — Entrez les valeurs des capteurs pour estimer le RUL</p>
            </div>""", unsafe_allow_html=True)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Accueil", key="btn_home3"):
                st.session_state.page_garde_shown = False
                st.rerun()

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
                        '<div style="margin-top:16px;background:white;border-radius:16px;padding:20px;box-shadow:0 4px 20px rgba(45,122,79,0.08);">'
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
        col_h, col_btn = st.columns([8, 1])
        with col_h:
            st.markdown("""<div class="page-header">
                <h1>Explainability — SHAP</h1>
                <p>Groupe OCP — Comprendre les decisions du modele</p>
            </div>""", unsafe_allow_html=True)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Accueil", key="btn_home4"):
                st.session_state.page_garde_shown = False
                st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Importance globale des features")
            img1 = os.path.join('..', 'models', 'shap_importance.png')
            if os.path.exists(img1): st.image(img1, use_container_width=True)
            else: st.warning("Generez d abord les images dans 03_SHAP.ipynb")
        with col2:
            st.markdown("### Impact des features sur le RUL")
            img2 = os.path.join('..', 'models', 'shap_beeswarm.png')
            if os.path.exists(img2): st.image(img2, use_container_width=True)
            else: st.warning("Generez d abord les images dans 03_SHAP.ipynb")

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
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a4d2e,#2d7a4f);border-radius:16px;padding:20px;margin-top:16px;">
            <div style="font-size:1rem;font-weight:600;margin-bottom:8px;color:white;">Conclusion SHAP</div>
            <div style="font-size:0.9rem;color:#a8d5b5;line-height:1.6;">
                Le modele PredictIQ s appuie principalement sur le <b style="color:white;">cycle de fonctionnement</b>
                et le <b style="color:white;">capteur s11</b> pour predire la duree de vie restante.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif page == "LSTM Deep Learning":
        col_h, col_btn = st.columns([8, 1])
        with col_h:
            st.markdown("""<div class="page-header">
                <h1>LSTM — Deep Learning</h1>
                <p>Groupe OCP — Prediction temporelle de la Duree de Vie Restante</p>
            </div>""", unsafe_allow_html=True)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Accueil", key="btn_home5"):
                st.session_state.page_garde_shown = False
                st.rerun()

        st.markdown("### Architecture du Modele LSTM")
        st.markdown("""
        <div style="display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;">
            <div style="flex:1;min-width:130px;background:white;border-radius:14px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(45,122,79,0.08);border-top:3px solid #2d7a4f;">
                <div style="font-size:0.8rem;color:#888;">Sequence</div><div style="font-size:1.6rem;font-weight:700;color:#1a4d2e;">30</div>
                <div style="font-size:0.75rem;color:#2d7a4f;">cycles contexte</div>
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
                <div style="font-size:0.8rem;color:#888;">Epochs</div><div style="font-size:1.6rem;font-weight:700;color:#c8860a;">50</div>
                <div style="font-size:0.75rem;color:#c8860a;">max</div>
            </div>
            <div style="flex:1;min-width:130px;background:white;border-radius:14px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(45,122,79,0.08);border-top:3px solid #c8860a;">
                <div style="font-size:0.8rem;color:#888;">Batch Size</div><div style="font-size:1.6rem;font-weight:700;color:#c8860a;">256</div>
                <div style="font-size:0.75rem;color:#c8860a;">observations</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Comparaison Random Forest vs LSTM")
        st.markdown("""
        <div style="background:white;border-radius:16px;padding:20px;box-shadow:0 4px 20px rgba(45,122,79,0.08);margin-bottom:20px;">
            <table style="width:100%;border-collapse:collapse;">
                <thead><tr style="background:#f5f0e8;">
                    <th style="padding:12px;text-align:left;color:#1a4d2e;">Critere</th>
                    <th style="padding:12px;text-align:center;color:#1a4d2e;">Random Forest</th>
                    <th style="padding:12px;text-align:center;color:#1a4d2e;">LSTM</th>
                </tr></thead>
                <tbody>
                    <tr style="border-bottom:1px solid #f0ebe0;">
                        <td style="padding:12px;font-weight:600;color:#555;">Type</td>
                        <td style="padding:12px;text-align:center;"><span class="badge badge-green">Machine Learning</span></td>
                        <td style="padding:12px;text-align:center;"><span class="badge badge-green">Deep Learning</span></td>
                    </tr>
                    <tr style="border-bottom:1px solid #f0ebe0;">
                        <td style="padding:12px;font-weight:600;color:#555;">Memoire temporelle</td>
                        <td style="padding:12px;text-align:center;color:#c0392b;">Non</td>
                        <td style="padding:12px;text-align:center;color:#2d7a4f;">Oui — 30 cycles</td>
                    </tr>
                    <tr style="border-bottom:1px solid #f0ebe0;">
                        <td style="padding:12px;font-weight:600;color:#555;">RMSE</td>
                        <td style="padding:12px;text-align:center;font-weight:700;">35.85</td>
                        <td style="padding:12px;text-align:center;font-weight:700;color:#2d7a4f;">14.92</td>
                    </tr>
                    <tr style="border-bottom:1px solid #f0ebe0;">
                        <td style="padding:12px;font-weight:600;color:#555;">MAE</td>
                        <td style="padding:12px;text-align:center;font-weight:700;">25.34</td>
                        <td style="padding:12px;text-align:center;font-weight:700;color:#2d7a4f;">11.16</td>
                    </tr>
                    <tr>
                        <td style="padding:12px;font-weight:600;color:#555;">R2</td>
                        <td style="padding:12px;text-align:center;font-weight:700;color:#2d7a4f;">0.72</td>
                        <td style="padding:12px;text-align:center;font-weight:700;color:#2d7a4f;">0.8613</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        img_curves = os.path.join('..', 'models', 'lstm_training_curves.png')
        img_pred   = os.path.join('..', 'models', 'lstm_prediction.png')
        if os.path.exists(img_curves):
            st.markdown("### Courbes d Apprentissage LSTM")
            st.image(img_curves, use_container_width=True)
        else:
            st.info("Executez d abord le notebook 04_LSTM.ipynb")
        if os.path.exists(img_pred):
            st.markdown("### Prediction vs Reel — LSTM")
            st.image(img_pred, use_container_width=True)

        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a4d2e,#2d7a4f);border-radius:16px;padding:20px;margin-top:16px;">
            <div style="font-size:1rem;font-weight:600;margin-bottom:8px;color:white;">Pourquoi LSTM ?</div>
            <div style="font-size:0.9rem;color:#a8d5b5;line-height:1.8;">
                Le LSTM analyse les <b style="color:white;">30 derniers cycles</b> pour comprendre la tendance de degradation.
                Contrairement au Random Forest, le LSTM capture la <b style="color:white;">dynamique temporelle</b>
                des capteurs — R2 passe de 0.72 a <b style="color:white;">0.8613</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif page == "Chatbot":
        col_h, col_btn = st.columns([8, 1])
        with col_h:
            st.markdown("""<div class="page-header">
                <h1>Chatbot PredictIQ</h1>
                <p>Groupe OCP — Posez vos questions sur la plateforme</p>
            </div>""", unsafe_allow_html=True)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Accueil", key="btn_home6"):
                st.session_state.page_garde_shown = False
                st.rerun()

        def get_response(question):
            q = question.lower().strip()
            if any(w in q for w in ["rul", "duree", "remaining", "vie", "cycles"]):
                return ("RUL — Remaining Useful Life<br><br>"
                        "Modeles : Random Forest et LSTM<br>"
                        "RF   : R2 = 0.72 — RMSE = 35.85<br>"
                        "LSTM : R2 = 0.8613 — RMSE = 14.92<br><br>"
                        "RUL sup 100 : Bon etat<br>"
                        "RUL 30 a 100 : Maintenance<br>"
                        "RUL inf 30 : Panne imminente")
            elif any(w in q for w in ["lstm", "deep", "learning", "neurone", "temporel"]):
                return ("LSTM — Long Short-Term Memory<br><br>"
                        "Architecture :<br>"
                        "Sequence : 30 cycles<br>"
                        "LSTM Layer 1 : 128 neurones<br>"
                        "LSTM Layer 2 : 64 neurones<br>"
                        "Dropout : 20 pourcent<br><br>"
                        "Resultats : R2=0.8613 — RMSE=14.92 — MAE=11.16<br>"
                        "Meilleur que Random Forest (R2=0.72)")
            elif any(w in q for w in ["shap", "expli", "important", "feature", "capteur"]):
                return ("SHAP — Explainable AI<br><br>"
                        "Top features :<br>"
                        "1. cycle — importance 38<br>"
                        "2. s11 — importance 11<br>"
                        "3. s9 — importance 5<br>"
                        "4. s12 — importance 4")
            elif any(w in q for w in ["dataset", "donnee", "nasa", "cmapss"]):
                return ("Dataset — NASA CMAPSS FD001<br><br>"
                        "20 631 observations<br>"
                        "100 moteurs — 21 capteurs<br>"
                        "6 capteurs supprimes<br>"
                        "Donnees reelles OCP en cours")
            elif any(w in q for w in ["modele", "model", "random", "forest", "algorithme"]):
                return ("Deux modeles utilises<br><br>"
                        "1. Random Forest — ML classique<br>"
                        "R2 = 0.72 — RMSE = 35.85 — MAE = 25.34<br><br>"
                        "2. LSTM — Deep Learning<br>"
                        "R2 = 0.8613 — RMSE = 14.92 — MAE = 11.16<br>"
                        "Memoire temporelle 30 cycles")
            elif any(w in q for w in ["ocp", "phosboucraa", "entreprise", "phosphate"]):
                return ("OCP Phosboucraa — Laayoune<br><br>"
                        "Leader mondial phosphates<br>"
                        "Plus de 20 000 employes<br>"
                        "Industrie 4.0")
            elif any(w in q for w in ["etat", "machine", "moteur", "status", "sante"]):
                return ("Etat du Systeme<br><br>"
                        "Sante : 92 pourcent<br>"
                        "RUL : 128 cycles<br>"
                        "Anomalie : 18.6 Normal<br>"
                        "Alertes : 2<br><br>"
                        "Temperature elevee — Turbine 3<br>"
                        "Vibration — Compresseur 2")
            elif any(w in q for w in ["maintenance", "panne", "recommandation"]):
                return ("Recommandations<br><br>"
                        "Priorite haute : Inspecter Turbine 3<br>"
                        "Priorite moyenne : Verifier Compresseur 2<br>"
                        "Priorite basse : Surveillance normale<br><br>"
                        "Reduction des couts de 30 pourcent")
            elif any(w in q for w in ["bonjour", "salut", "aide", "help", "bonsoir", "salam"]):
                return ("Bonjour — Chatbot PredictIQ<br><br>"
                        "Je reponds sur :<br>"
                        "RUL — Random Forest — LSTM<br>"
                        "SHAP — Dataset — Machines<br>"
                        "Maintenance — OCP<br><br>"
                        "Posez votre question")
            else:
                return ("Question non comprise.<br><br>"
                        "Essayez par exemple :<br>"
                        "Etat des machines<br>"
                        "Expliquez SHAP<br>"
                        "Expliquez LSTM<br>"
                        "Recommandations maintenance")

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    '<div style="display:flex;justify-content:flex-end;margin:8px 0;">'
                    '<div style="background:#1a4d2e;color:white;border-radius:18px 18px 4px 18px;'
                    'padding:12px 18px;max-width:70%;font-size:0.9rem;line-height:1.6;">'
                    + msg["content"] + '</div>'
                    '<div style="margin-left:8px;font-size:1.5rem;line-height:2;">U</div>'
                    '</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="display:flex;justify-content:flex-start;margin:8px 0;">'
                    '<div style="margin-right:8px;font-size:1.5rem;line-height:2;">B</div>'
                    '<div style="background:white;border:2px solid #e8e0d0;border-radius:18px 18px 18px 4px;'
                    'padding:12px 18px;max-width:75%;font-size:0.9rem;line-height:1.6;'
                    'box-shadow:0 2px 8px rgba(45,122,79,0.08);">'
                    + msg["content"] + '</div>'
                    '</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("q", placeholder="Posez votre question...",
                                       label_visibility="collapsed", key="chat_input")
        with col2:
            send = st.button("Envoyer", key="btn_send")

        st.markdown("**Questions rapides :**")
        qc1, qc2, qc3, qc4, qc5 = st.columns(5)
        quick = None
        with qc1:
            if st.button("Etat machines", key="q1"): quick = "etat des machines"
        with qc2:
            if st.button("SHAP", key="q2"): quick = "expliquez shap"
        with qc3:
            if st.button("LSTM", key="q3"): quick = "expliquez lstm"
        with qc4:
            if st.button("Modeles", key="q4"): quick = "quels modeles"
        with qc5:
            if st.button("Maintenance", key="q5"): quick = "recommandations maintenance"

        final_input = quick if quick else (user_input if send and user_input.strip() else None)
        if final_input:
            st.session_state.messages.append({"role": "user", "content": final_input})
            st.session_state.messages.append({"role": "assistant", "content": get_response(final_input)})
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Effacer la conversation", key="btn_clear"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Bonjour — Je suis le chatbot PredictIQ. Comment puis-je vous aider ?"}
            ]
            st.rerun()