import pandas as pd
import streamlit as st
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline

# Chargement du modèle depuis un fichier .pkl
model_path = 'XGBoostModel2.pkl'

def load_model(path):
    try:
        with open(path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

model = load_model(model_path)

if model is None:
    st.stop()

if isinstance(model, Pipeline):
    xgb_model = model.named_steps['xgbclassifier']
else:
    xgb_model = model

# Chargement des données depuis le CSV
dataset_path = 'X_train.csv'
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
    'DAYS_EMPLOYED': "Mois employés",
    'DAYS_BIRTH': "Âge en années"
}

# Mapping pour les colonnes binaires
gender_mapping = {0: 'Homme', 1: 'Femme'}
binary_mapping = {0: 'Non', 1: 'Oui'}

# Colonnes à afficher sur le dashboard
display_columns = list(column_mapping.keys())

def days_to_years(days):
    """Convertit les jours en années en prenant la valeur absolue des jours négatifs."""
    days = abs(days)
    years = days / 365
    return round(years, 2)

def days_to_months(days):
    """Convertit les jours en mois en prenant la valeur absolue des jours négatifs."""
    days = abs(days)
    months = days / 30
    return round(months, 2)

def get_client_data(client_id):
    client_data = df[df['SK_ID_CURR'] == client_id]
    
    if client_data.empty:
        return {}
    
    client_data = client_data.drop(columns=['SK_ID_CURR', 'TARGET']).iloc[0].to_dict()
    
    for feature in model.feature_names_in_:
        if feature not in client_data:
            client_data[feature] = 0.0
    
    return client_data

def predict(input_data):
    df = pd.DataFrame([input_data])
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    return predictions[0], probabilities[0]

def show_shap_explanation(input_data, xgb_model):
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(pd.DataFrame([input_data]))
        
        st.subheader("Explication Locale")
        plt.figure(figsize=(8, 4))
        shap.waterfall_plot(shap_values[0])
        st.pyplot(plt.gcf())
        
        st.subheader("Explication Globale")
        plt.figure(figsize=(8, 4))
        shap.summary_plot(shap_values, pd.DataFrame([input_data]), plot_type="bar", show=False)
        st.pyplot(plt.gcf())
        
    except Exception as e:
        st.error(f"Erreur avec SHAP: {e}")

def show_feature_importance(xgb_model):
    try:
        feature_importances = xgb_model.feature_importances_

        if hasattr(xgb_model, 'feature_names_in_'):
            feature_names = xgb_model.feature_names_in_
        else:
            feature_names = [f'Feature {i}' for i in range(len(feature_importances))]

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

def show_client_feature_distribution(client_data, df, display_columns):
    st.subheader("Comparaison des caractéristiques du client avec les autres clients")

    for feature in display_columns:
        if feature in client_data:
            plt.figure(figsize=(10, 4))
            
            if feature == 'CODE_GENDER':
                client_data[feature] = gender_mapping.get(client_data[feature], 'Inconnu')
                df[feature] = df[feature].map(gender_mapping)
            elif feature in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
                client_data[feature] = binary_mapping.get(client_data[feature], 'Inconnu')
                df[feature] = df[feature].map(binary_mapping)
            elif feature == 'DAYS_BIRTH':
                client_data[feature] = days_to_years(client_data[feature])
                df[feature] = df[feature].apply(days_to_years)
            elif feature == 'DAYS_EMPLOYED':
                client_data[feature] = days_to_months(client_data[feature])
                df[feature] = df[feature].apply(days_to_months)

            if feature == 'AMT_INCOME_TOTAL':
                sns.histplot(df[feature], kde=True, color='blue', label='Distribution globale')
                plt.axvline(client_data[feature], color='red', linestyle='--', label='Valeur du client')
                plt.title('Distribution de Revenu total')
                plt.xlabel('Revenu total')
                plt.ylabel('Fréquence')
                plt.legend()
            else:
                if df[feature].dtype == 'object':
                    sns.countplot(data=df, x=feature)
                    plt.title(f'Distribution de {column_mapping[feature]}')
                else:
                    sns.histplot(df[feature], kde=True, label='Distribution globale', color='blue')
                    plt.axvline(client_data[feature], color='red', linestyle='--', label='Valeur du client')
                    plt.title(f'Distribution de {column_mapping[feature]}')
                    plt.legend()
            
            st.pyplot(plt.gcf())

def show_bivariate_analysis(df, feature1='AMT_INCOME_TOTAL', feature2='CNT_CHILDREN'):
    st.subheader("Analyse Bivariée entre Revenu Total et Nombre d'Enfants")
    
    if feature1 in df.columns and feature2 in df.columns:
        st.write(f"Colonnes disponibles : {df.columns.tolist()}")  # Afficher les colonnes disponibles pour le débogage
        
        if pd.api.types.is_numeric_dtype(df[feature1]) and pd.api.types.is_numeric_dtype(df[feature2]):
            plt.figure(figsize=(10, 6))
            
            sns.scatterplot(data=df, x=feature1, y=feature2, alpha=0.6)
            
            plt.title(f'Analyse Bivariée entre {column_mapping.get(feature1, feature1)} et {column_mapping.get(feature2, feature2)}')
            plt.xlabel(column_mapping.get(feature1, feature1))
            plt.ylabel(column_mapping.get(feature2, feature2))
            
            st.pyplot(plt.gcf())
        else:
            st.error(f"Les colonnes {feature1} et {feature2} doivent être numériques.")
    else:
        st.error("Une ou les deux caractéristiques sélectionnées ne sont pas présentes dans le dataset.")

# Interface Streamlit
def main():
    st.set_page_config(page_title='Prédiction de Crédit', layout='wide', initial_sidebar_state='auto')

    #########
    # TITRE #
    #########
    st.header("Prédiction de Solvabilité")
    st.markdown("<h1 style='text-align: center;'>Explication de Solvabilité</h1>", unsafe_allow_html=True)

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
        
        # Mapper les valeurs après la prédiction
        if 'CODE_GENDER' in display_data.index:
            display_data.loc['CODE_GENDER', 'Valeur'] = gender_mapping.get(client_data.get('CODE_GENDER', 0), 'Inconnu')
        if 'FLAG_OWN_CAR' in display_data.index:
            display_data.loc['FLAG_OWN_CAR', 'Valeur'] = binary_mapping.get(client_data.get('FLAG_OWN_CAR', 0), 'Inconnu')
        if 'FLAG_OWN_REALTY' in display_data.index:
            display_data.loc['FLAG_OWN_REALTY', 'Valeur'] = binary_mapping.get(client_data.get('FLAG_OWN_REALTY', 0), 'Inconnu')
        if 'DAYS_BIRTH' in display_data.index:
            display_data.loc['DAYS_BIRTH', 'Valeur'] = days_to_years(client_data.get('DAYS_BIRTH', 0))
        if 'DAYS_EMPLOYED' in display_data.index:
            display_data.loc['DAYS_EMPLOYED', 'Valeur'] = days_to_months(client_data.get('DAYS_EMPLOYED', 0))

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

            progress_bar = st.sidebar.progress(0)
            for i in range(round(probability * 100)):
                progress_bar.progress(i + 1)

            if prediction == 1:
                st.sidebar.error("Crédit refusé !")
            elif prediction == 0:
                st.sidebar.success("Crédit accordé !")

            # Case à cocher pour afficher l'importance des caractéristiques
            if st.sidebar.checkbox('Afficher l\'importance des caractéristiques'):
                show_feature_importance(xgb_model)

            # Afficher l'explication de la prédiction
            show_shap_explanation(inputs, xgb_model)

        # Afficher la distribution des caractéristiques du client
        show_client_feature_distribution(client_data, df, display_columns)

        # Afficher l'analyse bi-variée entre 'Nombre' et 'AMT_INCOME_TOTAL'
        if 'CNT_CHILDREN' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            show_bivariate_analysis(df, 'CNT_CHILDREN', 'AMT_INCOME_TOTAL')

if __name__ == '__main__':
    main()
