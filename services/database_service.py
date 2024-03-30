import pyodbc
from ..config.config import Config

class DatabaseService:
    """Verwaltet Datenbankoperationen."""
    
    def __init__(self):
        """Initialisiert die Datenbankverbindung."""
        self.connection_string = Config.DB_CONNECTION_STRING
        self.connection = self._connect_to_database()
        
    def _connect_to_database(self):
        """Stellt eine Verbindung zur Datenbank her."""
        try:
            connection = pyodbc.connect(self.connection_string)
            print("Datenbankverbindung erfolgreich hergestellt.")
            return connection
        except Exception as e:
            print(f"Fehler bei der Verbindung zur Datenbank: {e}")
            return None
            
    def execute_query(self, query, params=None):
        """Führt eine SQL-Abfrage aus und gibt das Ergebnis zurück."""
        with self.connection.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            try:
                return cursor.fetchall()
            except pyodbc.ProgrammingError:
                # Für Abfragen, die keine Daten zurückgeben (z.B. INSERT, UPDATE, DELETE)
                return None
                
    def insert_predictions(self, predictions):
        """Fügt Vorhersagedaten in die Datenbank ein."""
        insert_query = """
        INSERT INTO Predictions (CustomerID, ProductID, PredictionScore, NextPurchaseDate)
        VALUES (?, ?, ?, ?)
        """
        with self.connection.cursor() as cursor:
            for prediction in predictions:
                cursor.execute(insert_query, prediction)
            self.connection.commit()

    def get_customers_for_prediction(self):
        """Holt eine Liste von Kunden für die Erstellung von Vorhersagen."""
        query = """
        SELECT CustomerID, AVG(Quantity) AS AvgQuantity, AVG(PricePerUnit) AS AvgPrice, MAX(PurchaseDate) AS LastPurchase
        FROM Transactions
        GROUP BY CustomerID
        HAVING MAX(PurchaseDate) < GETDATE() - 30  -- Beispiel: Kunden, die in den letzten 30 Tagen nichts gekauft haben
        """
        customers = self.execute_query(query)
        df_customers = pd.DataFrame(customers, columns=['CustomerID', 'AvgQuantity', 'AvgPrice', 'LastPurchase'])
        return df_customers
