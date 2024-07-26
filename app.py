import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn
import base64
import pickle
from io import StringIO

# Configurer l'URI de suivi MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Chargement du modèle complet (pipeline)
model_name = "XGBoostModel"
model_version = "1"  # Mettez à jour si vous avez plusieurs versions
model_uri = f"models:/{model_name}/{model_version}"

def load_model(uri):
    try:
        return mlflow.sklearn.load_model(uri)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle depuis MLflow: {e}")
        # Essayer de charger le modèle depuis un fichier local en cas d'échec
        try:
            st.warning("Tentative de chargement du modèle depuis un fichier local.")
            with open('XGBoostModel.pkl', 'rb') as model_file:
                return pickle.load(model_file)
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle depuis un fichier local: {e}")
            return None

model = load_model(model_uri)

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

# Fonction pour afficher les images base64 (pour les explications)
def show_image_from_base64(base64_image):
    image = base64.b64decode(base64_image)
    st.image(image)

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

    ############
    # PAGE PRINCIPALE #
    ############
    st.write(" ") # espace
    st.write(" ") # espace

    st.subheader('Explication de la Prédiction')

    # Affichage de l'explication de la prédiction (si applicable)
    if st.checkbox("Afficher l'explication de la prédiction"):
        feat_number = st.slider("Sélectionner le nombre de paramètres pour expliquer la prédiction", 1, 30, 10)

        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Explication Locale")
            base64_image = request_shap_waterfall_chart(client_id, feat_number)
            show_image_from_base64(base64_image)

        with col2:
            st.header("Explication Globale")
            base64_image = request_shap_waterfall_chart_global(feat_number)
            show_image_from_base64(base64_image)

if __name__ == '__main__':
    main()
