import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder

# Charger le modèle sauvegardé
model = joblib.load("bank_model.pkl")  

# Titre de l'application
st.title("Application de Prédictions de Comptes Bancaires")
st.write("Cette application prédit si un client a un compte bancaire ou non.")

st.subheader("Veuillez chargez de données pour la prédiction")
uploaded_file = st.file_uploader("Choisissez un fichier", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)
    
    st.write("Aperçu des données :", df.head()) 

if uploaded_file is not None:
    df_cleaned = df.dropna() # Suppression des valeurs manquantes
    df_cleaned = df_cleaned.drop_duplicates()  # Suppression des doublons
    df_cleaned = df_cleaned.drop(columns=['bank_account']) # Suppression de la colonne cible
    # Encodage des variables catégorielles   
    categorical_cols = [col for col in df_cleaned.select_dtypes(include=['object'])]  # Exclure la colonne 'country' de l'encodage
    le = LabelEncoder() 
    for col in categorical_cols:
        df_cleaned[col] = le.fit_transform(df_cleaned[col])
    st.write("Données après prétraitement :", df_cleaned.head())

    

# Bouton pour effectuer la prédiction
if st.button("Prédire"):            
    prediction = model.predict(df_cleaned) 
    result = pd.DataFrame(prediction, columns=["bank_account"])
    st.write("Données avec prédictions :", pd.concat([df_cleaned, result], axis=1, ignore_index=True).head())    
     

if st.button('Effacer les données'):
    st.rerun() 
    st.write("Données effacées.")

