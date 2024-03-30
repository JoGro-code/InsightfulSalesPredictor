# InsightfulSalesPredictor

## Overview

The InsightfulSalesPredictor is a sophisticated machine learning application designed to forecast future product purchases based on historical sales data. Utilizing a RandomForestRegressor model, this application predicts the next product a customer is likely to buy, along with a prediction score and the anticipated purchase date. Tailored for businesses similar to Würth, it aims to enhance sales strategies by providing insights into customer purchasing behaviors.

## Features

- **Data Processing:** Automated routines for cleaning and preparing sales data for model training and prediction.
- **Model Training:** A pipeline that includes training a RandomForestRegressor model on historical sales data.
- **Prediction Generation:** Ability to generate predictions for future purchases, including product IDs, prediction scores, and purchase dates.
- **Database Integration:** Services for interacting with an MSSQL database to fetch training data and store predictions.

## Getting Started

### Prerequisites

- Python 3.8+
- pip and venv
- Access to an MSSQL database server
- UnixODBC (for macOS and Linux users)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/jogro-code/InsightfulSalesPredictor.git
```

2. **Navigate to the project directory:**

```bash
cd InsightfulSalesPredictor
```

3. **Create and activate a virtual environment:**
   macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

4. **Install the required packages:**

```bash
pip install -r requirements.txt
```

### Configuration

1. **Database Connection:** Update the `.env` file with your database connection string.

```bash
DB_CONNECTION_STRING='DRIVER={SQL Server};SERVER=your_server;DATABASE=your_database;UID=your_username;PWD=your_password'
```

2. **Model Path:** Specify the path to save the trained model in the `config/config.py` file.

MODEL_PATH='models/my_model.joblib'

### Usage

**To train the model:**

```bash
python -m main train
```

**To generate predictions:**

```bash
python -m main predict
```

## Extending the Model

### Adding New Features

1. **Modify the Data Preprocessor:**

- Update `utils/data_preprocessor.py` to include new features during the data preparation phase. This might involve additional cleaning steps or feature engineering.

2. **Adjust the Model Pipeline:**

- If new features require specific transformations, modify the pipeline in `models/sales_prediction_model.py` to incorporate these changes.

### Adjusting Input Parameters

- To accommodate new input parameters for predictions, update the `predict` method in `services/prediction_service.py` to process additional data fields.

#### Adjusting Input Parameters - Example

Suppose you want to incorporate 'CustomerAge' and 'LastPurchaseCategory' as new input parameters to enhance the prediction accuracy. Here's how you could adjust the input parameters:

1.  **Update Data Preprocessing (`utils/data_preprocessor.py`):**
    Add 'CustomerAge' and 'LastPurchaseCategory' to the feature engineering section. For example, normalize 'CustomerAge' and apply OneHotEncoding to 'LastPurchaseCategory'.

    ```python
    # In utils/data_preprocessor.py
    def preprocess_data(df, training=True):
        ...
        if 'CustomerAge' in df.columns:
            df['NormalizedCustomerAge'] = (df['CustomerAge'] - df['CustomerAge'].mean()) / df['CustomerAge'].std()
            numerical_features.append('NormalizedCustomerAge')

        if 'LastPurchaseCategory' in df.columns and training:
            df = pd.get_dummies(df, columns=['LastPurchaseCategory'])
        ...
    ```

2.  **Modify Prediction Service (`services/prediction_service.py`):**
    Adjust the predict method to accept 'CustomerAge' and 'LastPurchaseCategory' as part of the input features.

        ```python
        # In services/prediction_service.py
        def predict(self, customer_features):
            ...
            # Ensure 'CustomerAge' and 'LastPurchaseCategory' are included in customer_features
            X = preprocess_data(customer_features, training=False)
            ...

        ```

### Modifying Output Parameters

- Modify the prediction output structure in `services/prediction_service.py` to include additional prediction outputs. Ensure that the database schema in `services/database_service.py` is updated accordingly to store any new output parameters.

#### Modifying Output Parameters - Example

Assuming you want to add 'EstimatedPurchaseValue' to the prediction output, follow these steps:

1.  **Adjust Model Prediction Output (`models/sales_prediction_model.py`):**

After generating the base prediction (e.g., ProductID, PredictionScore), append an estimate for 'EstimatedPurchaseValue'. This might require extending your model or applying additional business logic based on the predicted product.

    ```python
    # In models/sales_prediction_model.py
    class SalesPredictionModel(BaseEstimator, TransformerMixin):
        ...
        def predict(self, X):
            base_predictions = self.model.predict(X)
            # Example logic to calculate EstimatedPurchaseValue
            estimated_purchase_values = calculate_estimated_values(base_predictions)
            return np.hstack((base_predictions, estimated_purchase_values))

    ```

2.  **Update Database Schema and Service Prediction Output (`services/database_service.py`):**
    Modify the database schema to include a new column for 'EstimatedPurchaseValue'. Then, adjust insert_predictions method to save this new piece of information.

        ```sql
        --Example SQL to add a new column
        ALTER TABLE Predictions
        ADD EstimatedPurchaseValue DECIMAL(10, 2);
        ```

        ```python
        # In services/database_service.py
        def insert_predictions(self, predictions):
            ...
            insert_query = """
            INSERT INTO Predictions (CustomerID, ProductID, PredictionScore, NextPurchaseDate,      EstimatedPurchaseValue)
            VALUES (?, ?, ?, ?, ?)
            """
            ...

        ```

This example demonstrates how to incorporate new input parameters ('CustomerAge' and 'LastPurchaseCategory') into the prediction process and extend the prediction output to include 'EstimatedPurchaseValue'. It covers modifications in data preprocessing, the prediction service, and the model itself, as well as adjustments to the database schema to store new output parameters. The specifics of calculating 'EstimatedPurchaseValue' would depend on additional business logic or model output.

## Contributing

Contributions are welcome! Please feel free to submit pull requests with new features, improvements, or bug fixes.

## License

Distributed under the MIT License. See `LICENSE` for more information.
