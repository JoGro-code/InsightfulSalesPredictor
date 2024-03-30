from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
import joblib

class SalesPredictionModel(BaseEstimator, TransformerMixin):
    """
    Ein Modell zur Vorhersage der nächsten ProductID, PredictionScore und des Kaufdatums
    basierend auf Kunden- und Kaufdaten.
    """
    
    def __init__(self):
        """
        Initialisiert das SalesPredictionModel mit einem RandomForestRegressor
        und einem vordefinierten Preprocessing-Pipeline.
        """
        # Definieren der Vorverarbeitung für numerische und kategoriale Spalten
        numeric_features = ['Quantity', 'PricePerUnit']
        categorical_features = ['BranchCode', 'CustomerValue']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ])
        
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
    def fit(self, X, y):
        """
        Trainiert das Modell mit den gegebenen Daten.
        
        Args:
            X (pandas.DataFrame): Die Eingabedaten für das Training.
            y (pandas.Series): Die Zielvariable für das Training.
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Generiert Vorhersagen für die gegebenen Daten.
        
        Args:
            X (pandas.DataFrame): Die Daten, für die Vorhersagen generiert werden sollen.
            
        Returns:
            numpy.ndarray: Die generierten Vorhersagen.
        """
        predictions = self.model.predict(X)
        return predictions
    
    def save_model(self, path='models/sales_prediction_model.joblib'):
        """
        Speichert das trainierte Modell an einem angegebenen Pfad.
        
        Args:
            path (str): Der Pfad, an dem das Modell gespeichert werden soll.
        """
        joblib.dump(self.model, path)
        
    def load_model(self, path='models/sales_prediction_model.joblib'):
        """
        Lädt ein trainiertes Modell von einem angegebenen Pfad.
        
        Args:
            path (str): Der Pfad, von dem das Modell geladen werden soll.
        """
        self.model = joblib.load(path)
