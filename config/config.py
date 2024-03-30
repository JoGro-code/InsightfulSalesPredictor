from dotenv import load_dotenv
import os

# Lädt Umgebungsvariablen aus der .env-Datei
load_dotenv()

class Config:
    """Konfigurationsklasse für die Anwendung."""
    
    # Datenbankverbindungsstring
    DB_CONNECTION_STRING = os.getenv('DB_CONNECTION_STRING', 'DefaultConnectionString')
    
    # Pfad zum Speichern des trainierten Modells
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/sales_prediction_model.joblib')
