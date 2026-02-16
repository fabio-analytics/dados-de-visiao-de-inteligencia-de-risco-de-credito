import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="VisionData Pro | Analytics", page_icon="💎", layout="wide")

# --- 2. CARREGAMENTO DOS DADOS E MODELO (COM CACHE) ---
@st.cache_resource
def load_model_objects():
    try:
        model = joblib.load('modelo_random_forest.pkl')
        cols = joblib.load('colunas_modelo.pkl')
        return model, cols
    except:
        return None, None

model, model_cols = load_model_objects()

# --- 3. INTERFACE E ESTILO ---
# Título e Sidebar
st.sidebar.markdown(f"<h1 style='text-align: left; color: #00E5FF;'>💎 VisionData</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.info("Dashboard Interativo de Risco de Crédito")

# Título Principal
st.markdown("""
    <h1 style='text-align: center; color: white;'>
        🛡️ VisionData Pro <span style='color: #00E5FF;'>Credit Analytics</span>
    </h1>
    <p style='text-align: center; color: gray;'>Engine de Decisão Financeira Baseada em Machine Learning (Random Forest)</p>
    <hr>
""", unsafe_allow_html=True)

# --- 4. ÁREA DE SIMULAÇÃO (INPUTS) ---
st.markdown("### 🤖 Simulador de Aprovação de Crédito")

col_in, col_res = st.columns([2, 1])

with col_in:
    st.markdown("#### Perfil do Cliente")
    s_fico = st.slider("Score FICO (Histórico de Crédito)", 300, 850, 700, help="Abaixo de 640 é considerado risco alto.")
    
    c1, c2 = st.columns(2)
    with c1: 
        s_dti = st.slider("DTI % (Dívida sobre Renda)", 0.0, 40.0, 15.0, help="Acima de 20% indica alto endividamento.")
    with c2: 
        s_inc = st.number_input("Log da Renda Anual", 5.0, 15.0, 10.5)
    
    s_int = st.number_input("Taxa de Juros Proposta (%)", 5.0, 25.0, 12.0) / 100

with col_res:
    st.markdown("<br>", unsafe_allow_html=True)
    # Botão Grande
    calcular = st.button("CALCULAR RISCO 🚀", use_container_width=True)

    if calcular:
        if model:
            # --- 5. LÓGICA HÍBRIDA (BANCO RIGOROSO) ---
            
            # Passo 1: Prepara os dados para a IA
            input_data = pd.DataFrame(0, index=[0], columns=model_cols)
            input_data['fico'] = s_fico
            input_data['dti'] = s_dti
            input_data['int.rate'] = s_int
            input_data['log.annual.inc'] = s_inc
            
            # Valores padrão para colunas que não pedimos no front
            input_data['credit.policy'] = 1
            input_data['installment'] = 300
            input_data['days.with.cr.line'] = 4000
            input_data['revol.bal'] = 10000
            input_data['revol.util'] = 50
            
            # Passo 2: A IA calcula a probabilidade base
            prob_ia = model.predict_proba(input_data)[0][1]
            
            # Passo 3: REGRAS DE NEGÓCIO (O "Pulo do Gato" para o Demo)
            # Aqui forçamos o risco para cima se o perfil for ruim, independente da IA
            prob_final = prob_ia
            motivo = "Baseado no histórico do LendingClub."
            
            override = False
            if s_fico < 640:
                prob_final = 0.85 # Força 85% de risco
                motivo = "Score FICO abaixo do mínimo aceitável (640)."
                override = True
            elif s_dti > 20:
                prob_final = 0.75 # Força 75% de risco
                motivo = "Comprometimento de renda (DTI) muito alto."
                override = True
            
            # Passo 4: EXIBIÇÃO DO RESULTADO
            # Régua de aprovação: Risco deve ser menor que 15%
            if prob_final > 0.15: # Régua super rigorosa
                st.error("❌ CRÉDITO NEGADO")
                st.metric(
                    label="Probabilidade de Calote", 
                    value=f"{prob_final*100:.1f}%", 
                    delta="Alto Risco", 
                    delta_color="inverse"
                )
                st.warning(f"⚠️ Motivo da Recusa: {motivo}")
            else:
                st.success("✅ CRÉDITO APROVADO")
                st.metric(
                    label="Probabilidade de Calote", 
                    value=f"{prob_final*100:.1f}%", 
                    delta="Seguro", 
                    delta_color="normal"
                )
                st.caption("Cliente apto para concessão de crédito.")

        else:
            st.error("Erro: Modelo não carregado. Verifique os arquivos .pkl")
