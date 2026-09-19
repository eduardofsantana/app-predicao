
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

# Configuração da página
st.set_page_config(page_title='Data App - Predição de Imóveis', layout='wide')

@st.cache_resource
def load_model():
    return joblib.load('modelo.pkl')

# Carregar dados para visualização (Boston Dataset)
@st.cache_data
def load_data():
    return pd.read_csv('https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/d4332a3056f44e1a1dec9600a31f21c8_boston.csv')

model = load_model()
data = load_data()

# TÍTULO E INTRODUÇÃO (CENTRO)
st.title('🏠 Data App - Prevendo valores de imóveis')
st.markdown('Este app exibe a análise exploratória e predições de preços do dataset Boston House Prices.')

# --- SEÇÃO 1: ANÁLISE EXPLORATÓRIA (CENTRO) ---
st.header('🔍 Análise Exploratória dos Dados')
defaultcols = ['RM', 'PTRATIO', 'CRIM', 'MEDV']
cols = st.multiselect('Atributos para visualização:', data.columns.tolist(), default=defaultcols)

st.subheader('Amostra dos Dados')
st.dataframe(data[cols].head(10))

st.subheader('📊 Distribuição de Preços')
faixa_valores = st.slider('Filtrar faixa de preço (MEDV)', float(data.MEDV.min()), float(data.MEDV.max()), (10.0, 40.0))
dados_filtrados = data[data['MEDV'].between(faixa_valores[0], faixa_valores[1])]

fig = px.histogram(dados_filtrados, x='MEDV', nbins=30, title='Distribuição de Preços Selecionados', color_discrete_sequence=['#4EA8DE'])
st.plotly_chart(fig, use_container_width=True)

# --- SEÇÃO 2: PREDIÇÃO (BARRA LATERAL) ---
st.sidebar.header('⚙️ Atributos para Predição')
crim = st.sidebar.number_input('CRIM (Taxa crime)', value=float(data.CRIM.mean()))
indus = st.sidebar.number_input('INDUS (Indústria)', value=float(data.INDUS.mean()))
chas_label = st.sidebar.selectbox('Faz limite com rio?', ('Sim', 'Não'))
chas = 1 if chas_label == 'Sim' else 0
nox = st.sidebar.number_input('NOX (Ox. Nítrico)', value=float(data.NOX.mean()))
rm = st.sidebar.number_input('RM (Quartos)', value=6.0)
pratio = st.sidebar.number_input('PTRATIO (Alunos/Prof)', value=float(data.PTRATIO.mean()))

if st.sidebar.button('🔮 Realizar Predição'):
    entrada = [[crim, indus, chas, nox, rm, pratio]]
    predicao = model.predict(entrada)[0]
    st.sidebar.success(f'Valor estimado: US$ {round(predicao * 10, 2)} mil')
    st.balloons()
