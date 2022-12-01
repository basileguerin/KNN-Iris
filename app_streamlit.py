import streamlit as st
import numpy as np
import joblib

#récupération du modele
model = joblib.load('final_model')
scaler = joblib.load('minmaxscaler')

def espece(model, sepalLength, sepalWidth, petalLength, petalWidth):
    """Renvoie à quelle espèce d'iris appartient la fleur dont les dimensions
    du sépal et du pétal ont été données par l'utilisateur"""
    
    especes = {0:'setosa', 1:'versicolor', 2:'virignica'}
    x = np.array([sepalLength, sepalWidth, petalLength, petalWidth]).reshape(1,4)
    x_scale = scaler.transform(x)
    return especes[model.predict(x_scale)[0]]

with st.form('add values', clear_on_submit=True):
    sepalLength = st.slider("sepal.length", 0, 10)
    sepalWidth = st.slider("sepal.width", 0, 10)
    petalLength = st.slider("petal.length", 0, 10)
    petalWidth = st.slider("petal.width", 0, 10)
    submitted = st.form_submit_button('Submit')
    if submitted:
        st.write("""L'iris est de l'espèce :""", espece(model, sepalLength, sepalWidth, petalLength, petalWidth))

st.image('iris.jpg')
