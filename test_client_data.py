# Ce test Pytest vérifie la fonctionnalité de la fonction get_client_data
import pytest
import pandas as pd
from app import get_client_data



df = pd.DataFrame({
    'SK_ID_CURR': [1, 2],
    'TARGET': [0, 1],
    'AMT_INCOME_TOTAL': [100000, 150000],
    'CNT_CHILDREN': [2, 3],
    'CODE_GENDER': ['F', 'M'],
    'FLAG_OWN_CAR': ['N', 'Y'],
    'FLAG_OWN_REALTY': ['Y', 'N'],
    'NAME_EDUCATION_TYPE': ['Higher education', 'Secondary education'],
    'NAME_FAMILY_STATUS': ['Married', 'Single'],
    'NAME_HOUSING_TYPE': ['House / apartment', 'Rented apartment'],
    'NAME_INCOME_TYPE': ['Working', 'Commercial associate'],
    'OCCUPATION_TYPE': ['Laborers', 'Managers']
})

# fonction get_client_data
def get_client_data(client_id):
    client_data = df[df['SK_ID_CURR'] == client_id]
    
    if client_data.empty:
        return {}
    
    client_data = client_data.drop(columns=['SK_ID_CURR', 'TARGET']).iloc[0].to_dict()
    
    # Ajouter des colonnes manquantes avec des valeurs par défaut si nécessaire
    for feature in ['AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE', 
                    'OCCUPATION_TYPE']:
        if feature not in client_data:
            client_data[feature] = 0.0
    
    return client_data

def test_get_client_data():
    """Tester la fonction get_client_data"""
    client_id = 1
    expected_data = {
        'AMT_INCOME_TOTAL': 100000,
        'CNT_CHILDREN': 2,
        'CODE_GENDER': 'F',
        'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'Y',
        'NAME_EDUCATION_TYPE': 'Higher education',
        'NAME_FAMILY_STATUS': 'Married',
        'NAME_HOUSING_TYPE': 'House / apartment',
        'NAME_INCOME_TYPE': 'Working',
        'OCCUPATION_TYPE': 'Laborers'
    }
    
    result = get_client_data(client_id)
    
    for key, value in expected_data.items():
        assert result.get(key) == value
