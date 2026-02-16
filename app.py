import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="VisionData Pro | Analytics", page_icon="💎", layout="wide")

# --- 2. CARREGAMENTO DOS DADOS E MODELO ---
@st.cache_resource
def load_data():
    try:
        # Carrega o Dataset para fazer os gráficos
        df = pd.read_csv('loan_data.csv')
        # Carrega o Modelo
        model = joblib.load('modelo_random_forest.pkl')
        cols = joblib.load('colunas_modelo.pkl')
        return df, model, cols
    except Exception as e:
        return None, None, None

df, model, model_cols = load_data()

# --- 3. BARRA LATERAL (INPUTS) ---
st.sidebar.markdown(f"<h1 style='text-align: left; color: #9B59B6;'>💎 VisionData 2.0</h1>", unsafe_allow_html=True)
st.sidebar.info("Simulador de Crédito com IA + Regras de Bloqueio")
st.sidebar.markdown("---")

st.sidebar.header("📝 Perfil do Cliente")
# Inputs movidos para a lateral para dar espaço aos gráficos
s_fico = st.sidebar.slider("Score FICO", 300, 850, 630, help="Abaixo de 660 reprova.")
s_dti = st.sidebar.slider("DTI (Dívida/Renda %)", 0.0, 40.0, 20.0)
s_inc = st.sidebar.number_input("Renda Anual (Log)", 5.0, 15.0, 10.5)
s_int = st.sidebar.number_input("Taxa de Juros (%)", 5.0, 25.0, 12.0) / 100

st.sidebar.markdown("---")
calcular = st.sidebar.button("CALCULAR RISCO 🚀", type="primary")

# --- 4. ÁREA PRINCIPAL (RESULTADOS E GRÁFICOS) ---

st.markdown("""
    <h2 style='text-align: center;'>
        🛡️ Dashboard de Análise de Risco
    </h2>
    <hr>
""", unsafe_allow_html=True)

# Só mostra o resultado se clicar no botão
if calcular:
    
    # --- LÓGICA DE APROVAÇÃO (HARD RULES) ---
    decision = "APROVADO"
    motivo = ""
    probabilidade = 0.10
    classe_risco = "Baixo"
    cor_box = "success"

    # 1. Regra FICO Baixo
    if s_fico < 660:
        decision = "NEGADO"
        motivo = "Score FICO abaixo do mínimo aceitável (660)."
        probabilidade = 0.88
        classe_risco = "Altíssimo"
        cor_box = "error"
    
    # 2. Regra DTI Alto
    elif s_dti > 22:
        decision = "NEGADO"
        motivo = "Comprometimento de renda (DTI) excessivo."
        probabilidade = 0.75
        classe_risco = "Alto"
        cor_box = "error"
    
    # 3. Consulta IA (Se passar das regras acima)
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
        
        if prob_ia > 0.25:
            decision = "NEGADO"
            motivo = "IA detectou padrão histórico de calote."
            probabilidade = prob_ia
            classe_risco = "Moderado-Alto"
            cor_box = "error"
        else:
            decision = "APROVADO"
            motivo = "Perfil sólido segundo histórico."
            probabilidade = prob_ia
            classe_risco = "Seguro"
            cor_box = "success"

    # --- EXIBIÇÃO DOS RESULTADOS (KPIs) ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Decisão Final", decision, delta=classe_risco, delta_color="inverse" if decision=="NEGADO" else "normal")
    c2.metric("Probabilidade de Calote", f"{probabilidade*100:.1f}%")
    c3.metric("Score do Cliente", s_fico)

    if decision == "NEGADO":
        st.error(f"❌ **CRÉDITO NEGADO:** {motivo}")
    else:
        st.success(f"✅ **CRÉDITO APROVADO:** {motivo}")
    
    st.markdown("---")

# --- 5. OS GRÁFICOS (VISUALIZAÇÃO DE DADOS) ---
# Se o arquivo csv estiver carregado, mostra os gráficos
if df is not None:
    st.markdown("### 📊 Comparativo com a Base de Dados")
    
    col_g1, col_g2 = st.columns(2)

    # Gráfico 1: Onde o cliente está no FICO?
    with col_g1:
        st.markdown("**Distribuição de FICO Score**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x='fico', hue='not.fully.paid', bins=25, kde=True, palette="viridis", ax=ax1)
        # Linha vermelha mostrando o cliente
        plt.axvline(s_fico, color='red', linestyle='--', linewidth=3, label='Você Está Aqui')
        plt.legend()
        st.pyplot(fig1)
        st.caption("A linha vermelha mostra seu Score comparado com todos os clientes do banco.")

    # Gráfico 2: Relação DTI x Juros
    with col_g2:
        st.markdown("**Análise de Endividamento (DTI)**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        # Filtra um pouco para não ficar pesado
        sns.scatterplot(data=df.sample(500), x='dti', y='int.rate', hue='not.fully.paid', palette="coolwarm", ax=ax2)
        plt.scatter(s_dti, s_int, color='red', s=200, marker='X', label='Você') # Marca o cliente com um X
        st.pyplot(fig2)
        st.caption("O 'X' vermelho mostra sua posição de Dívida vs Juros.")

else:
    st.warning("Arquivo 'loan_data.csv' não encontrado. Suba o CSV para ver os gráficos!")
