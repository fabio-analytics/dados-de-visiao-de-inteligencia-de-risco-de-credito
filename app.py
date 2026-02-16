import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="VisionData Pro | Analytics", page_icon="💎", layout="wide")

# --- 2. CARREGAMENTO ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('loan_data.csv')
        model = joblib.load('modelo_random_forest.pkl')
        cols = joblib.load('colunas_modelo.pkl')
        return df, model, cols
    except:
        return None, None, None

df, model, model_cols = load_data()

# --- 3. CABEÇALHO ---
st.markdown("""
    <h1 style='text-align: center; color: white;'>
        🛡️ VisionData Pro <span style='color: #00E5FF;'>Credit Analytics</span>
    </h1>
    <p style='text-align: center; color: gray;'>Machine Learning Credit Scoring System</p>
    <hr>
""", unsafe_allow_html=True)

# --- 4. ÁREA DE INPUTS (LAYOUT CENTRAL) ---
st.subheader("📝 Simulação de Cliente")

col1, col2, col3, col4 = st.columns(4)

with col1:
    s_fico = st.number_input("Score FICO", 300, 850, 630, help="Abaixo de 640 reprova automaticamente.")
with col2:
    s_dti = st.number_input("DTI (Dívida %)", 0.0, 40.0, 20.0)
with col3:
    s_inc = st.number_input("Renda Log (Ex: 10.5)", 5.0, 15.0, 10.5)
with col4:
    s_int = st.number_input("Taxa de Juros (%)", 5.0, 25.0, 12.0) / 100

# Botão Grande Centralizado
col_vazia_esq, col_btn, col_vazia_dir = st.columns([1, 2, 1])
with col_btn:
    calcular = st.button("CALCULAR RISCO 🚀", use_container_width=True, type="primary")

# --- 5. LÓGICA DE CÁLCULO (HÍBRIDA) ---
if calcular:
    st.markdown("---")
    
    # REGRAS DE BLOQUEIO (HARD RULES)
    decision = "APROVADO"
    motivo = ""
    probabilidade = 0.10
    cor = "success"
    
    # Regra 1: FICO Baixo
    if s_fico < 640:
        decision = "NEGADO"
        motivo = "Score FICO abaixo do mínimo aceitável (640)."
        probabilidade = 0.85
        cor = "error"
    
    # Regra 2: DTI Alto
    elif s_dti > 22:
        decision = "NEGADO"
        motivo = "Comprometimento de renda (DTI) excessivo."
        probabilidade = 0.78
        cor = "error"
        
    # Regra 3: IA (Só chama se passar das regras acima)
    elif model:
        # Prepara dados
        input_data = pd.DataFrame(0, index=[0], columns=model_cols)
        input_data['fico'] = s_fico
        input_data['dti'] = s_dti
        input_data['int.rate'] = s_int
        input_data['log.annual.inc'] = s_inc
        # Padrões
        input_data['credit.policy'] = 1
        input_data['installment'] = 300
        input_data['days.with.cr.line'] = 4000
        input_data['revol.bal'] = 10000
        input_data['revol.util'] = 50
        
        prob_ia = model.predict_proba(input_data)[0][1]
        
        if prob_ia > 0.30: # Régua da IA
            decision = "NEGADO"
            motivo = "Modelo de IA identificou alto risco."
            probabilidade = prob_ia
            cor = "error"
        else:
            probabilidade = prob_ia

    # EXIBIÇÃO DO RESULTADO
    c_res1, c_res2 = st.columns([1, 3])
    with c_res1:
        st.metric("Resultado Final", decision, delta="Risco Calculado", delta_color="inverse" if decision=="NEGADO" else "normal")
    with c_res2:
        if decision == "NEGADO":
            st.error(f"❌ **CRÉDITO REPROVADO:** {motivo}")
        else:
            st.success("✅ **CRÉDITO APROVADO:** Cliente com bom perfil pagador.")
            
    st.progress(int(probabilidade * 100), text=f"Probabilidade de Calote: {probabilidade*100:.1f}%")

# --- 6. GRÁFICOS (VISUAL CLÁSSICO DE 4 COLUNAS) ---
st.markdown("---")
mostrar_graficos = st.checkbox("📊 Exibir Análise Exploratória (Dashboard)", value=True)

if mostrar_graficos and df is not None:
    st.markdown("### Panorama da Carteira de Crédito")
    
    # Layout de 2 linhas e 2 colunas para os gráficos
    g_col1, g_col2 = st.columns(2)
    
    # Linha 1
    with g_col1:
        st.caption("Distribuição de FICO Scores")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x='fico', hue='not.fully.paid', bins=20, palette='viridis', ax=ax1)
        plt.axvline(s_fico, color='red', linestyle='--', label='Você')
        st.pyplot(fig1)

    with g_col2:
        st.caption("Taxa de Juros por Dívida")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df.sample(300), x='dti', y='int.rate', hue='not.fully.paid', palette='coolwarm', ax=ax2)
        plt.scatter(s_dti, s_int, color='red', s=100, marker='X', label='Você')
        st.pyplot(fig2)

    g_col3, g_col4 = st.columns(2)
    
    # Linha 2
    with g_col3:
        st.caption("Finalidade dos Empréstimos")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, y='purpose', palette='magma', ax=ax3)
        st.pyplot(fig3)

    with g_col4:
        st.caption("Renda Anual (Log)")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='not.fully.paid', y='log.annual.inc', palette='pastel', ax=ax4)
        plt.axhline(s_inc, color='red', linestyle='--', label='Você')
        st.pyplot(fig4)

elif mostrar_graficos:
    st.warning("Carregando gráficos... (Verifique se loan_data.csv está no GitHub)")
