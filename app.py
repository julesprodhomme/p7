import pandas as pd
import streamlit as st
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
from imblearn.pipeline import Pipeline

# Chargement du modèle depuis un fichier .pkl
model_path = 'XGBoostModel.pkl'

def load_model(path):
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

model = load_model(model_path)

# Vérifiez si le modèle a été chargé avec succès
if model is None:
    st.stop()

# Vérifier si le modèle est un pipeline et extraire le modèle XGBoost
if isinstance(model, Pipeline):
    xgb_model = model.named_steps['xgbclassifier']  # Assurez-vous que le nom correspond à votre pipeline
else:
    xgb_model = model

# Chargement des données depuis le CSV
dataset_path = 'X_train_smote.csv'
try:
    df = pd.read_csv(dataset_path)
except Exception as e:
    st.error(f"Erreur lors du chargement des données: {e}")
    st.stop()

X_train_smote = df.drop(columns=['TARGET'])  # Assurez-vous de retirer la colonne TARGET si elle est présente
y_train_smote = df['TARGET']

# Fonction pour obtenir les données du client en fonction de l'identifiant
def get_client_data(client_id):
    client_data = df[df['SK_ID_CURR'] == client_id]
    
    if client_data.empty:
        return {}
    
    client_data = client_data.drop(columns=['SK_ID_CURR', 'TARGET']).iloc[0].to_dict()
    
    # Ajouter des colonnes manquantes avec des valeurs par défaut si nécessaire
    for feature in model.feature_names_in_:
        if feature not in client_data:
            client_data[feature] = 0.0
    
    return client_data

# Fonction pour faire des prédictions
def predict(input_data):
    df = pd.DataFrame([input_data])
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    return predictions[0], probabilities[0]

# Fonction pour afficher les explications SHAP
def show_shap_explanation(X_train, y_train, model):
    try:
        # Extraire le modèle XGBoost du pipeline
        xgb_model = model.named_steps['xgbclassifier'] if isinstance(model, Pipeline) else model
        
        # Calculer les valeurs SHAP
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(X_train)
        
        # Convertir les valeurs SHAP en DataFrame
        shap_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': np.abs(shap_values.values).mean(axis=0)
        })
        
        # Trier les caractéristiques par importance
        shap_df = shap_df.sort_values(by='Importance', ascending=False)
        
        # Afficher les 10 caractéristiques les plus importantes
        top_10_shap = shap_df.head(10)
        
        # Visualisation des importances SHAP
        st.subheader("Top 10 des caractéristiques les plus importantes selon SHAP")
        plt.figure(figsize=(12, 8))
        plt.barh(top_10_shap['Feature'], top_10_shap['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Top 10 des caractéristiques les plus importantes selon SHAP')
        plt.gca().invert_yaxis()
        st.pyplot(plt.gcf())
        
    except Exception as e:
        st.error(f"Erreur avec SHAP: {e}")

# Interface Streamlit
def main():
    st.set_page_config(page_title='Prédiction de Crédit', page_icon=':credit_card:', layout='wide', initial_sidebar_state='auto')

    #########
    # TITRE #
    #########
    st.header("Prédiction de Solvabilité")
    st.markdown("<h1 style='text-align: center; border: 2px solid black; padding: 10px; background-color: #cccccc; border-radius: 10px;'> Prédiction et Explication de Solvabilité</h1>", unsafe_allow_html=True)

    ############
    # SIDEBAR #
    ###########
    st.sidebar.header("Sélection du Client")

    # Sélection de l'identifiant du client
    client_id = st.sidebar.selectbox("**Sélectionner l'identifiant du client**", options=df['SK_ID_CURR'].unique())

    if client_id:
        # Obtenez les données du client
        client_data = get_client_data(client_id)
        
        # Filtrer les données du client pour ne garder que les colonnes désirées
        filtered_data = {key: client_data.get(key, None) for key in display_columns}
        
        # Préparer les données pour le tableau avec les nouveaux noms
        display_data = pd.DataFrame.from_dict(filtered_data, orient='index', columns=['Valeur'])
        display_data.index = display_data.index.map(column_mapping)
        
        # Afficher les données du client sous forme de tableau avec défilement horizontal
        st.sidebar.subheader('Données du client')
        st.sidebar.dataframe(display_data, use_container_width=True)
        
        # Préparer les données pour la prédiction
        inputs = {feature: client_data.get(feature, 0.0) for feature in model.feature_names_in_}
        
        # Bouton pour faire la prédiction
        if st.sidebar.button("Faire une prédiction"):
            prediction, probability = predict(inputs)
            st.sidebar.subheader('Résultats de la Prédiction')
            st.sidebar.write(f"Prédiction: {'Classe 1' if prediction == 1 else 'Classe 0'}")
            st.sidebar.write(f"Probabilité de Classe 1: {probability:.2f}")

            # Barre de progression affichant le % de chance de remboursement
            progress_bar = st.sidebar.progress(0)
            for i in range(round(probability * 100)):
                progress_bar.progress(i + 1)

            # Affichage de la prédiction
            if prediction == 1:
                st.sidebar.error("Crédit refusé !")
            elif prediction == 0:
                st.sidebar.success("Crédit accordé !")

            # Afficher l'explication des caractéristiques importantes
            show_shap_explanation(X_train_smote, y_train_smote, model)

if __name__ == '__main__':
    main()
