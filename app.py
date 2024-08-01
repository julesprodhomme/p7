import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn
import base64
import shap
import matplotlib.pyplot as plt


# Chargement du modèle complet (pipeline)
model_name = "XGBoostModel.pkl"


# Vérifiez si le modèle a été chargé avec succès
if model is None:
    st.stop()

# Chargement des données depuis le CSV
dataset_path = 'X_train_smote.csv'
try:
    df = pd.read_csv(dataset_path)
except Exception as e:
    st.error(f"Erreur lors du chargement des données: {e}")
    st.stop()

# Dictionnaire de mappage des noms de colonnes
column_mapping = {
    'AMT_INCOME_TOTAL': "Revenu total",
    'CNT_CHILDREN': "Nombre d'enfants",
    'CODE_GENDER': "Genre",
    'FLAG_OWN_CAR': "Possède un véhicule",
    'FLAG_OWN_REALTY': "Propriétaire immobilier",
    'NAME_EDUCATION_TYPE': "Niveau académique",
    'NAME_FAMILY_STATUS': "Statut familial",
    'NAME_HOUSING_TYPE': "Type de logement",
    'NAME_INCOME_TYPE': "Type de revenu",
    'OCCUPATION_TYPE': "Emploi"
}

# Colonnes à afficher sur le dashboard
display_columns = list(column_mapping.keys())

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
def show_shap_explanation(input_data, model):
    try:
        # Si le modèle est un pipeline, utiliser l'étape XGBoost
        if hasattr(model, 'named_steps') and 'xgbclassifier' in model.named_steps:
            xgb_model = model.named_steps['xgbclassifier']
        else:
            xgb_model = model

        # Créer un explainer SHAP pour le modèle XGBoost
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(pd.DataFrame([input_data]))

        st.subheader("Explication Locale")
        plt.figure(figsize=(8, 4))  # Réduire la taille du graphique
        shap.waterfall_plot(shap_values[0], show=False)
        st.pyplot(plt.gcf())

        st.subheader("Explication Globale")
        plt.figure(figsize=(8, 4))  # Réduire la taille du graphique
        shap.summary_plot(shap_values, pd.DataFrame([input_data]), plot_type="bar", show=False)
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
        inputs = {
