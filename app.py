import pandas as pd
import streamlit as st
import pickle
import shap
import matplotlib.pyplot as plt
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
#if isinstance(model, Pipeline):
    #xgb_model = model.named_steps['xgbclassifier']  # Assurez-vous que le nom correspond à votre pipeline
#else:
xgb_model = model

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
        #st.header(type(feature))
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
def show_shap_explanation(input_data, xgb_model):
    st.header("1")
    try:
        # Créer un explainer SHAP pour le modèle XGBoost
        explainer = shap.TreeExplainer(xgb_model)
        st.header("2")
        
        # Préparer les données pour SHAP
        st.header(xgb_model.feature_names_in_)
        #input_df = pd.DataFrame([input_data], columns=xgb_model.feature_names_in_)
        input_df = pd.DataFrame(input_data)

        st.header("3")
        
        # Calculer les valeurs SHAP
        shap_values = explainer(input_df)
        st.header("4")
        
        st.subheader("Explication Locale")
        plt.figure(figsize=(8, 4))  # Réduire la taille du graphique
        shap.waterfall_plot(shap_values[0])
        st.pyplot(plt.gcf())
        st.header("5")
        
        st.subheader("Explication Globale")
        plt.figure(figsize=(8, 4))  # Réduire la taille du graphique
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        st.pyplot(plt.gcf())
        st.header("6")
    except Exception as e:
        st.error(f"Erreur avec SHAP: {e}")
    st.header("Final")
        

# Fonction pour afficher l'importance des caractéristiques
def show_feature_importance(xgb_model):
    st.header("A")
    try:
        # Extraire les importances des caractéristiques
        feature_importances = xgb_model.feature_importances_

        # Récupérer les noms des caractéristiques
        if hasattr(xgb_model, 'feature_names_in_'):
            feature_names = xgb_model.feature_names_in_
        else:
            feature_names = [f'Feature {i}' for i in range(len(feature_importances))]

        # Créer un DataFrame pour les importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        st.subheader("Importance des Caractéristiques")
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Importance des Caractéristiques selon XGBoost')
        plt.gca().invert_yaxis()
        st.pyplot(plt.gcf())
    except Exception as e:
        st.error(f"Erreur avec l'importance des caractéristiques: {e}")
    st.header("Ending")

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

            # Case à cocher pour afficher l'importance des caractéristiques
            if st.sidebar.checkbox('Afficher l\'importance des caractéristiques'):
                show_feature_importance(xgb_model)

            # Afficher l'explication de la prédiction
            show_shap_explanation(inputs, xgb_model)

if __name__ == '__main__':
    main()
