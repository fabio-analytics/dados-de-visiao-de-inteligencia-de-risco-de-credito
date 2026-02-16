import streamlit as st
import pandas as pd
import joblib

# --- 1. CONFIGURAÇÃO DA PÁGINA (TÍTULO NOVO) ---
st.set_page_config(page_title="VisionData 2.0 | Rules Engine", page_icon="🔒", layout="wide")

# --- 2. CARREGAMENTO DOS DADOS ---
@st.cache_resource
def load_model_objects():
    try:
        model = joblib.load('modelo_random_forest.pkl')
        cols = joblib.load('colunas_modelo.pkl')
        return model, cols
    except:
        return None, None

model, model_cols = load_model_objects()

# --- 3. INTERFACE VISUAL (MUDEI A COR DO TÍTULO PARA ROXO) ---
st.sidebar.markdown(f"<h1 style='text-align: left; color: #9B59B6;'>🔒 VisionData 2.0</h1>", unsafe_allow_html=True)
st.sidebar.info("Versão Atualizada: Regras de Bloqueio Ativas")
st.sidebar.markdown("---")

st.markdown("""
    <h1 style='text-align: center; color: white;'>
        🛡️ VisionData Pro <span style='color: #9B59B6;'>Credit Rules</span>
    </h1>
    <p style='text-align: center;'>Sistema Híbrido: Inteligência Artificial + Regras de Bloqueio Bancário</p>
    <hr>
""", unsafe_allow_html=True)

# --- 4. ÁREA DE SIMULAÇÃO ---
col_in, col_res = st.columns([2, 1])

with col_in:
    st.markdown("### 🏦 Simulador de Crédito")
    
    # SLIDERS (Aumentei o FICO padrão para 700 para começar aprovando)
    s_fico = st.slider("Score FICO (Se baixar de 660 reprova)", 300, 850, 700)
    
    c1, c2 = st.columns(2)
    with c1: s_dti = st.slider("DTI % (Dívida/Renda)", 0.0, 40.0, 15.0)
    with c2: s_inc = st.number_input("Renda Logarítmica", 5.0, 15.0, 10.5)
    
    s_int = st.number_input("Taxa de Juros %", 5.0, 25.0, 12.0) / 100

with col_res:
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # BOTÃO
    if st.button("CALCULAR RISCO 🚀", use_container_width=True):
        
        # --- LÓGICA DE BLOQUEIO (HARD RULES) ---
        decision = "APROVADO"
        motivo = ""
        probabilidade = 0.10 # Começa baixo (10%)

        # REGRA 1: FICO BAIXO (Aumentei a régua)
        if s_fico < 660:
            decision = "NEGADO"
            motivo = "Score FICO muito baixo (Risco de Inadimplência)."
            probabilidade = 0.88 # Força 88% de risco

        # REGRA 2: DTI ALTO
        elif s_dti > 22:
            decision = "NEGADO"
            motivo = "Cliente com muitas dívidas (DTI Alto)."
            probabilidade = 0.75 # Força 75% de risco

        # SE PASSAR DAS REGRAS, CHAMA A IA (Só pra compor, se tiver modelo)
        elif model:
             # Prepara dados pra IA
             input_data = pd.DataFrame(0, index=[0], columns=model_cols)
             input_data['fico'] = s_fico
             input_data['dti'] = s_dti
             input_data['int.rate'] = s_int
             input_data['log.annual.inc'] = s_inc
             
             # Valores padrão
             input_data['credit.policy'] = 1
             input_data['installment'] = 300
             input_data['days.with.cr.line'] = 4000
             input_data['revol.bal'] = 10000
             input_data['revol.util'] = 50
             
             # Pega a probabilidade real da IA
             prob_ia = model.predict_proba(input_data)[0][1]
             
             # Se a IA achar ruim, também nega
             if prob_ia > 0.25:
                 decision = "NEGADO"
                 motivo = "Modelo de IA detectou padrão de risco."
                 probabilidade = prob_ia
             else:
                 # Se a IA achar bom, mantém o aprovado, mas atualiza a prob
                 probabilidade = prob_ia

        # --- EXIBIÇÃO FINAL ---
        if decision == "NEGADO":
            st.error("❌ CRÉDITO NEGADO")
            st.metric("Probabilidade de Calote", f"{probabilidade*100:.1f}%", delta="Alto Risco", delta_color="inverse")
            st.write(f"**Motivo:** {motivo}")
        else:
            st.success("✅ CRÉDITO APROVADO")
            st.metric("Probabilidade de Calote", f"{probabilidade*100:.1f}%", delta="Seguro", delta_color="normal")
            st.write("**Parecer:** Cliente apto para crédito.")
