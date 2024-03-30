from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
from .database_service import DatabaseService
from ..utils.data_preprocessor import preprocess_data
from ..config.config import Config

class PredictionService:
    """Verwaltet das Trainieren des Modells und die Erstellung von Vorhersagen."""
    
    def __init__(self):
        self.db_service = DatabaseService()
        self.model_path = Config.MODEL_PATH
        self.model = None
        
    def train_model(self):
        """Lädt Trainingsdaten, führt Vorverarbeitung durch, trainiert das Modell und speichert es."""
        # Laden der Trainingsdaten
        query = """
        SELECT CustomerID, ProductID, BranchCode, CustomerValue, Utilization, Quantity, PricePerUnit, PurchaseDate
        FROM TrainingData
        """
        training_data = self.db_service.execute_query(query)
        df = pd.DataFrame(training_data, columns=['CustomerID', 'ProductID', 'BranchCode', 'CustomerValue', 'Utilization', 'Quantity', 'PricePerUnit', 'PurchaseDate'])
        
        # Vorverarbeitung der Daten
        X, y = preprocess_data(df)
        
        # Aufteilung der Daten in Trainings- und Testdatensätze
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Training des Modells
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Speichern des trainierten Modells
        joblib.dump(self.model, self.model_path)
        print("Modell erfolgreich trainiert und gespeichert unter:", self.model_path)
        
    def load_model(self):
        """Lädt das trainierte Modell."""
        if self.model is None:
            self.model = joblib.load(self.model_path)
            print("Modell geladen von:", self.model_path)
        
    def predict(self, customer_features):
        """Generiert eine Vorhersage für gegebene Kundendaten."""
        # Stellen Sie sicher, dass das Modell geladen ist
        self.load_model()
        
        # Umwandlung der Kundendaten in DataFrame und Vorverarbeitung
        df = pd.DataFrame([customer_features], columns=['BranchCode', 'CustomerValue', 'Quantity', 'PricePerUnit'])
        X = preprocess_data(df, training=False)
        
        # Erstellen der Vorhersage
        prediction = self.model.predict(X)
        return prediction
