
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

# Configuração inicial
st.set_page_config(page_title='Predição de Imóveis', layout='wide')

# Carregar modelo e dados
@st.cache_resource
def load_assets():
    model = joblib.load('modelo.pkl')
    return model

model = load_assets()

st.title('🏠 Data App - Prevendo valores de imóveis')
st.markdown('Solução de ML com Random Forest para o dataset Boston House Prices.')

# Sidebar para inputs
st.sidebar.header('⚙️ Atributos do Imóvel')
crim = st.sidebar.number_input('CRIM (Taxa crime)', value=0.36)
indus = st.sidebar.number_input('INDUS (Indústria)', value=11.0)
chas_label = st.sidebar.selectbox('Limite com Rio?', ('Sim', 'Não'))
chas = 1 if chas_label == 'Sim' else 0
nox = st.sidebar.number_input('NOX (Ox. Nítrico)', value=0.55)
rm = st.sidebar.number_input('RM (Quartos)', value=6.0)
pratio = st.sidebar.number_input('PTRATIO (Alunos/Prof)', value=18.0)

if st.sidebar.button('🔮 Realizar Predição'):
    entrada = [[crim, indus, chas, nox, rm, pratio]]
    predicao = model.predict(entrada)[0]
    st.success(f'### Valor estimado: US$ {round(predicao * 10, 2)} mil')
