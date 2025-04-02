import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder



# Charger le modèle pré-entraîné
model = joblib.load("bank_model.pkl")


# Titre de l'application
st.title("Application de Prédictions de Comptes Bancaires")
st.write("Cette application prédit si un client est susceptible de posseder un compte bancaire ou non.")



# Choix du mode de prédiction
mode = st.radio("Choisissez le mode de prédiction", ["Fichier", "Entrée manuelle"])

if mode == "Fichier":
    st.subheader("Téléchargement de données")
    st.write("Téléchargez le fichier  contenant les données à prédire.")
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
        df_cleaned = df_cleaned.drop(columns=['bank_account','uniqueid','country']) # Suppression de la colonne cible
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
            st.write("Resultat des prédictions :", pd.concat([df_cleaned, result], axis=1, ignore_index=True).head())  
     

        if st.button('Effacer les données'):
            st.rerun() 
            st.write("Données effacées.")

else:
    
    st.subheader("Entrée manuelle des données")

    # Entrée utilisateur avec toutes tes variables
    
    year = st.number_input("Année", min_value=1900, max_value=2025)
    household_size = st.number_input("Taille du foyer", min_value=0, max_value=20)
    age_of_respondent = st.number_input("Âge du répondant", min_value=18, max_value=100)
    location_type = st.selectbox("Type de localisation", ["Urban", "Rural"])
    cellphone_access = st.selectbox("Accès à un téléphone portable", ["Yes", "No"])
    gender = st.selectbox("Genre", ["Male", "Female"])
    relationship_with_head = st.selectbox("Lien avec le chef de famille", ["Head", "Spouse", "Child", "Other"])
    education_level = st.selectbox("Niveau d'éducation", ["Primary", "Secondary", "Tertiary"])
    job_type = st.selectbox("Type d'emploi", ["Self-employed", "Government", "Private", "Informal"])
    marital_status = st.selectbox("État civil", ["Single", "Married", "Divorced", "Widowed"])
    # Encodage des valeurs catégorielles avec mapping
    location_mapping = {"Urban": 0, "Rural": 1}
    cellphone_mapping = {"Yes": 1, "No": 0}
    gender_mapping = {"Male": 0, "Female": 1}
    relationship_mapping = {"Head": 0, "Spouse": 1, "Child": 2, "Other": 3}
    education_mapping = {"Primary": 0, "Secondary": 1, "Tertiary": 2}
    job_mapping = {"Self-employed": 0, "Government": 1, "Private": 2, "Informal": 3}
    marital_status_mapping = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
    # Création du tableau de données pour la prédiction
    input_data = np.array([[ year, household_size, age_of_respondent, 
                            location_mapping[location_type], cellphone_mapping[cellphone_access], 
                            gender_mapping[gender], relationship_mapping[relationship_with_head],
                            education_mapping[education_level], job_mapping[job_type],
                            marital_status_mapping[marital_status]]])

    # Prédiction à partir des entrées utilisateur
    if st.button("Prédire"):
        prediction = model.predict(input_data)
        st.success(f"Prédiction : {'Compte bancaire' if prediction[0] == 1 else 'Pas de compte bancaire'}")