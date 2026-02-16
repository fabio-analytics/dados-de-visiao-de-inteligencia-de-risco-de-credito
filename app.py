import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import matplotlib.colors as mcolors

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="VisionData Pro | Analytics", page_icon="💎", layout="wide")

# --- 2. CONTROLE DE TEMA ---
# Título na Barra Lateral
st.sidebar.markdown(f"<h1 style='text-align: left; color: #00E5FF;'>💎 VisionData</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Personalização")

# Botão de Troca de Tema
modo_escuro = st.sidebar.toggle("Ativar Modo Dark Premium", value=True)

# --- 3. DEFINIÇÃO DAS PALETAS (CORRIGIDO PARA NÃO DAR ERRO) ---
if modo_escuro:
    # TEMA DARK (Estilo "Deep Space Glass")
    THEME = {
        "bg_hex": "#0F2027", # Cor sólida para uso no Matplotlib (Correção do Erro)
        "bg_gradient": "linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%)",
        "sidebar_bg": "#0B151E",
        "card_bg": "rgba(255, 255, 255, 0.05)", # Vidro bem transparente
        "card_border": "rgba(255, 255, 255, 0.1)", # Borda sutil
        "text_primary": "#FFFFFF",
        "text_secondary": "#B0BEC5",
        "accent": "#00E5FF",           # Ciano Neon
        "accent_secondary": "#7B1FA2", # Roxo Neon
        "shadow": "0 8px 32px 0 rgba(0, 0, 0, 0.5)",
        "grid_color": "#37474F",
        "success": "#00E676",
        "danger": "#FF1744"
    }
    plt.style.use('dark_background')
else:
    # TEMA LIGHT (Estilo "Platinum Mirror")
    THEME = {
        "bg_hex": "#FFFFFF", # Cor sólida para Matplotlib
        "bg_gradient": "linear-gradient(135deg, #E0EAFC 0%, #CFDEF3 100%)",
        "sidebar_bg": "#FFFFFF",
        "card_bg": "rgba(255, 255, 255, 0.6)", # Vidro leitoso
        "card_border": "rgba(255, 255, 255, 0.8)",
        "text_primary": "#1A237E",      # Azul Marinho Profundo
        "text_secondary": "#455A64",
        "accent": "#0D47A1",            # Azul Royal
        "accent_secondary": "#FF6F00",
        "shadow": "0 8px 32px 0 rgba(31, 38, 135, 0.1)",
        "grid_color": "#B0BEC5",
        "success": "#2E7D32",
        "danger": "#C62828"
    }
    plt.style.use('default')

# Atualização Dinâmica do Matplotlib
plt.rcParams.update({
    "figure.facecolor": "(0,0,0,0)",
    "axes.facecolor": "(0,0,0,0)",
    "savefig.facecolor": "(0,0,0,0)",
    "text.color": THEME["text_primary"],
    "axes.labelcolor": THEME["text_primary"],
    "xtick.color": THEME["text_primary"],
    "ytick.color": THEME["text_primary"],
    "grid.color": THEME["grid_color"],
    "axes.edgecolor": THEME["grid_color"]
})

# --- 4. CSS AVANÇADO (EFEITO ESPELHADO + CORREÇÃO BOTÃO) ---
st.markdown(f"""
    <style>
    /* Fundo com Gradiente Premium */
    .stApp {{
        background: {THEME['bg_gradient']};
        background-attachment: fixed;
        color: {THEME['text_primary']};
    }}

    /* --- CORREÇÃO DA BARRA LATERAL (BOTÃO VISÍVEL) --- */
    section[data-testid="stSidebar"] {{
        background-color: {THEME['sidebar_bg']} !important;
        border-right: 1px solid {THEME['card_border']};
    }}
    /* Força a cor de TODOS os textos na sidebar, incluindo o botão */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] div {{
        color: {THEME['text_primary']} !important;
    }}
    
    /* --- EFEITO DE VIDRO/ESPELHO NOS CARTÕES --- */
    div[data-testid="stMetric"], .glass-card {{
        background: {THEME['card_bg']};
        backdrop-filter: blur(16px); /* O Segredo do vidro fosco */
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {THEME['card_border']};
        border-radius: 16px;
        box-shadow: {THEME['shadow']};
        padding: 20px;
    }}

    /* Títulos e Métricas */
    h1, h2, h3, h4 {{ color: {THEME['text_primary']} !important; font-family: 'Segoe UI', sans-serif; }}
    div[data-testid="stMetricLabel"] {{ color: {THEME['text_secondary']} !important; }}
    div[data-testid="stMetricValue"] {{ color: {THEME['accent']} !important; text-shadow: 0 0 10px rgba(0,229,255,0.3); }}

    /* Título Principal Gradiente */
    .main-title {{
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, {THEME['accent']}, {THEME['accent_secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }}
    
    /* Cabeçalho dos Cartões */
    .card-header {{
        color: {THEME['accent']};
        font-weight: 700;
        font-size: 1.1rem;
        border-bottom: 1px solid {THEME['card_border']};
        padding-bottom: 10px;
        margin-bottom: 15px;
        letter-spacing: 1px;
    }}
    
    /* Ajuste de Margem */
    .block-container {{ padding-top: 2rem; padding-bottom: 5rem; }}
    
    /* Estilo do Botão do Simulador */
    .stButton > button {{
        background: linear-gradient(90deg, {THEME['accent']}, {THEME['accent_secondary']});
        border: none;
        color: #fff;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: 0.3s;
    }}
    .stButton > button:hover {{
        transform: scale(1.02);
        box-shadow: 0 0 20px {THEME['accent']};
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 5. CARREGAMENTO DE DADOS ---
@st.cache_resource
def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_path, 'loan_data.csv'))
    path_model = os.path.join(base_path, 'modelo_random_forest.pkl')
    path_cols = os.path.join(base_path, 'colunas_modelo.pkl')
    model = joblib.load(path_model) if os.path.exists(path_model) else None
    model_cols = joblib.load(path_cols) if os.path.exists(path_cols) else None
    return df, model, model_cols

df, model, model_cols = load_data()

# --- 6. BARRA LATERAL ---
st.sidebar.markdown("### 🔍 Filtros Inteligentes")
all_purposes = df['purpose'].unique()
filtro_proposito = st.sidebar.multiselect("Finalidade:", options=all_purposes, default=all_purposes)
df_filtrado = df[df['purpose'].isin(filtro_proposito)]

st.sidebar.markdown("<br>", unsafe_allow_html=True)
# Cartãozinho na Sidebar
st.sidebar.markdown(f"""
<div style='background: {THEME['card_bg']}; padding: 15px; border-radius: 10px; border: 1px solid {THEME['card_border']}'>
    <small style='color: {THEME['text_secondary']}'>Contratos Filtrados</small><br>
    <strong style='color: {THEME['accent']}; font-size: 1.5em;'>{len(df_filtrado):,}</strong>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.link_button("🌐 VisionDataPro.com", "http://visiondatapro.com")

# --- 7. ÁREA PRINCIPAL ---
st.markdown('<div class="main-title">Executive Risk Overview</div>', unsafe_allow_html=True)
st.markdown(f"<p style='color: {THEME['text_secondary']}; font-size: 1.1rem;'>Dashboard Estratégico de Análise de Crédito com Inteligência Artificial.</p>", unsafe_allow_html=True)

st.markdown("---")

# KPIs
k1, k2, k3, k4 = st.columns(4)
inad_filtrada = df_filtrado['not.fully.paid'].mean() * 100
delta_inad = inad_filtrada - (df['not.fully.paid'].mean() * 100)

with k1: st.metric("Volume Analisado", f"{len(df_filtrado):,}")
with k2: st.metric("Juros Médios", f"{df_filtrado['int.rate'].mean()*100:.2f}%")
with k3: st.metric("Taxa de Risco", f"{inad_filtrada:.2f}%", delta=f"{delta_inad:.2f}% vs Global", delta_color="inverse")
with k4: st.metric("Score FICO", f"{int(df_filtrado['fico'].mean())}")

# --- 8. GRÁFICOS PREMIUM ---
def clean_plot(ax):
    sns.despine(ax=ax, left=True, bottom=True)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, color=THEME['grid_color'], alpha=0.3)
    ax.tick_params(axis='both', which='both', length=0)
    return ax

st.markdown("<br>", unsafe_allow_html=True)
g1, g2 = st.columns(2)

with g1:
    st.markdown('<div class="glass-card"><div class="card-header">📊 Distribuição FICO x Política</div>', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    colors = {0: THEME['danger'], 1: THEME['success']} if modo_escuro else {0: '#E53935', 1: '#43A047'}
    
    sns.histplot(data=df_filtrado, x='fico', hue='credit.policy', 
                 palette=colors, element="step", fill=True, alpha=0.6, linewidth=0, ax=ax1)
    ax1.set_xlabel("Score FICO")
    ax1.set_ylabel("Quantidade")
    clean_plot(ax1)
    st.pyplot(fig1)
    st.markdown('</div>', unsafe_allow_html=True)

with g2:
    st.markdown('<div class="glass-card"><div class="card-header">🎯 Risco por Finalidade</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    order = df_filtrado['purpose'].value_counts().index
    paleta = "rocket" if modo_escuro else "Blues_r"
    
    sns.countplot(data=df_filtrado, y='purpose', hue='not.fully.paid', 
                  order=order, palette=paleta, ax=ax2)
    ax2.set_xlabel("Volume")
    ax2.set_ylabel("")
    ax2.legend(title="Inadimplente?", loc='lower right', frameon=False, labelcolor=THEME['text_primary'])
    clean_plot(ax2)
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

g3, g4 = st.columns(2)

with g3:
    st.markdown('<div class="glass-card"><div class="card-header">📈 Tendência: Juros vs. Score</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    colors_scat = {0: THEME['accent'], 1: THEME['danger']} if modo_escuro else {0: '#1E88E5', 1: '#D32F2F'}
    
    sns.scatterplot(data=df_filtrado, x='fico', y='int.rate', hue='not.fully.paid',
                    palette=colors_scat, alpha=0.7, s=60, linewidth=0, ax=ax3)
    ax3.set_xlabel("Score FICO")
    ax3.set_ylabel("Taxa de Juros")
    ax3.legend(frameon=False, labelcolor=THEME['text_primary'])
    clean_plot(ax3)
    st.pyplot(fig3)
    st.markdown('</div>', unsafe_allow_html=True)

with g4:
    st.markdown('<div class="glass-card"><div class="card-header">🌡️ Mapa de Calor (Correlação)</div>', unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    cols = ['int.rate', 'fico', 'dti', 'installment', 'log.annual.inc']
    corr = df_filtrado[cols].corr()
    
    # CORREÇÃO DO ERRO ANTERIOR: Usando 'bg_hex' em vez de 'bg_main'
    cmap = mcolors.LinearSegmentedColormap.from_list("", [THEME['bg_hex'], THEME['accent'], "#ffffff"])
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, 
                cbar_kws={'shrink': .8}, linewidths=0.5, linecolor=THEME['bg_hex'], ax=ax4)
    clean_plot(ax4)
    cbar = ax4.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=THEME['text_primary'])
    st.pyplot(fig4)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 9. SIMULADOR IA (Efeito Vidro Neon) ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div class="glass-card" style="border: 1px solid {THEME['accent']}; box-shadow: 0 0 20px rgba(0, 229, 255, 0.1);">
    <div class="card-header" style="color: {THEME['accent']}; border: none; font-size: 1.5rem;">
        🤖 Simulador de Decisão (AI Engine)
    </div>
""", unsafe_allow_html=True)

col_in, col_res = st.columns([2, 1])

with col_in:
    s_fico = st.slider("Score FICO", 300, 850, 700)
    c1, c2 = st.columns(2)
    with c1: s_dti = st.slider("DTI %", 0.0, 40.0, 15.0)
    with c2: s_inc = st.number_input("Renda Log", 5.0, 15.0, 10.5)
    s_int = st.number_input("Juros Propostos %", 5.0, 25.0, 12.0) / 100

with col_res:
    st.markdown("<br><br>", unsafe_allow_html=True)
   # --- BOTÃO DE CÁLCULO (COM NOVA CALIBRAGEM) ---
    if st.button("CALCULAR RISCO 🚀", use_container_width=True):
        if model:
            # 1. Cria o dataframe com os inputs do usuário
            input_data = pd.DataFrame(0, index=[0], columns=model_cols)
            input_data['fico'] = s_fico
            input_data['dti'] = s_dti
            input_data['int.rate'] = s_int
            input_data['log.annual.inc'] = s_inc
            
            # Preenche valores padrão para o que não pedimos (para o modelo não quebrar)
            input_data['credit.policy'] = 1
            input_data['installment'] = 300  # Valor médio
            input_data['days.with.cr.line'] = 4500 # Valor médio
            input_data['revol.bal'] = 15000 # Valor médio
            input_data['revol.util'] = 50 # Valor médio
            
            # 2. Pega a probabilidade (O Cérebro da IA)
            # prob_calote é a chance de NÃO pagar (Classe 1)
            prob_calote = model.predict_proba(input_data)[0][1]
            
            # 3. EXIBE O RESULTADO
            # Se a chance de calote for maior que 20%, o Banco nega! (Régua Alta)
            if prob_calote > 0.20:
                st.error(f"❌ CRÉDITO NEGADO (Alto Risco)")
                st.metric(label="Probabilidade de Calote (Inadimplência)", value=f"{prob_calote*100:.1f}%", delta="-Alto Risco", delta_color="inverse")
                st.write("Motivo: O perfil financeiro apresenta indicadores de instabilidade baseados no histórico do LendingClub.")
            else:
                st.success(f"✅ CRÉDITO APROVADO")
                st.metric(label="Probabilidade de Calote (Inadimplência)", value=f"{prob_calote*100:.1f}%", delta="Risco Baixo", delta_color="normal")
                st.write("Parabéns! O perfil foi classificado como seguro para concessão de crédito.")
        else:
            st.error("Erro: Modelo não carregado.")
