import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df, training=True):
    """
    Bereitet die Eingabedaten für das Modell vor, einschließlich Feature Engineering,
    One-Hot-Encoding für kategoriale Variablen und Skalierung für numerische Variablen.
    
    Args:
        df (pandas.DataFrame): Die Eingabedaten als DataFrame.
        training (bool): Gibt an, ob es sich um Trainingsdaten handelt. Wenn False,
                         werden bestimmte Schritte wie das Fitten des Encoders übersprungen.
    
    Returns:
        X (pandas.DataFrame): Die vorverarbeiteten Daten, bereit für das Modell.
        y (pandas.Series, optional): Die Zielvariable, nur zurückgegeben, wenn `training=True`.
    """
    # Identifizieren kategorischer und numerischer Features
    categorical_features = ['BranchCode']
    numerical_features = ['CustomerValue', 'Quantity', 'PricePerUnit']

    # Ergänzung für Vorhersagedaten
    if not training:
        # Prüfung, ob Vorhersagedaten spezifische Features enthalten
        if 'AvgQuantity' in df.columns and 'AvgPrice' in df.columns and 'LastPurchase' in df.columns:
            # Umwandlung des 'LastPurchase' Datums in Tage seit dem letzten Kauf
            df['DaysSinceLastPurchase'] = (pd.to_datetime('now') - pd.to_datetime(df['LastPurchase'])).dt.days
            numerical_features.extend(['AvgQuantity', 'AvgPrice', 'DaysSinceLastPurchase'])
            # Entfernen der ursprünglichen Features aus der Liste, um Duplikate zu vermeiden
            numerical_features = list(set(numerical_features) - {'Quantity', 'PricePerUnit'})

    # Feature Engineering für Trainingsdaten
    if 'PurchaseDate' in df.columns:
        df['DaysSincePurchase'] = (pd.to_datetime('now') - pd.to_datetime(df['PurchaseDate'])).dt.days
        numerical_features.append('DaysSincePurchase')
    
    # Pipeline für kategoriale Features
    categorical_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Pipeline für numerische Features
    numerical_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Zusammenführen der Pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
    
    # Anpassung der Pipelines auf die Daten
    if training:
        X = preprocessor.fit_transform(df)
    else:
        X = preprocessor.transform(df)
    
    # Zielvariable separieren, wenn im Trainingsmodus
    if training and 'Utilization' in df.columns:
        y = df['Utilization']
        return X, y
    else:
        return X
