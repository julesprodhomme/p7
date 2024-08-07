# tests/test_load_csv.py
import pytest
import pandas as pd
from app import load_csv

def test_load_csv():
    # Créez un DataFrame de test
    test_data = {
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
    }
    
    test_df = pd.DataFrame(test_data) 
    
    # Sauvegardez le DataFrame dans un fichier CSV temporaire
    test_csv_path = 'test_data.csv'
    test_df.to_csv(test_csv_path, index=False)
    
    # Chargez le CSV avec la fonction à tester
    loaded_df = load_csv(test_csv_path)
    
    # Vérifiez que les données chargées sont les mêmes que les données d'origine
    pd.testing.assert_frame_equal(loaded_df, test_df)
    
    # Supprimez le fichier CSV temporaire après le test
    import os
    os.remove(test_csv_path)
