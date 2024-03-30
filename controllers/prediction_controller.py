from datetime import datetime
from ..services.database_service import DatabaseService
from ..services.prediction_service import PredictionService
from ..utils.data_preprocessor import preprocess_data
import pandas as pd

def run_training():
    """
    Koordiniert den Trainingsprozess des Vorhersagemodells.
    """
    db_service = DatabaseService()
    prediction_service = PredictionService()

    # Laden der Trainingsdaten aus der Datenbank
    query = """
    SELECT CustomerID, ProductID, BranchCode, CustomerValue, Utilization, Quantity, PricePerUnit, PurchaseDate
    FROM TrainingData
    """
    training_data = db_service.execute_query(query)
    df = pd.DataFrame(training_data, columns=['CustomerID', 'ProductID', 'BranchCode', 'CustomerValue', 'Utilization', 'Quantity', 'PricePerUnit', 'PurchaseDate'])
    
    # Daten vorbereiten
    X, y = preprocess_data(df)
    
    # Modell trainieren
    prediction_service.train_model(X, y)
    print("Training abgeschlossen.")

def run_prediction():
    """
    Koordiniert den Vorhersageprozess und speichert die Ergebnisse in der Datenbank.
    """
    db_service = DatabaseService()
    prediction_service = PredictionService()
    
    # Angenommen, wir haben eine Methode, um neue oder existierende Kunden für Vorhersagen zu identifizieren
    customers_for_prediction = db_service.get_customers_for_prediction()
    
    predictions = []
    for customer in customers_for_prediction:
        # Daten vorbereiten (angenommen, customer enthält bereits die erforderlichen Features)
        X = preprocess_data(customer, training=False)
        
        # Vorhersage generieren
        predicted_utilization, predicted_productID, predicted_purchase_date = prediction_service.predict(X)
        
        # Vorhersage für die Speicherung vorbereiten
        predictions.append((customer['CustomerID'], predicted_productID, predicted_utilization, predicted_purchase_date))
    
    # Vorhersagen in der Datenbank speichern
    db_service.insert_predictions(predictions)
    print("Vorhersagen gespeichert.")

