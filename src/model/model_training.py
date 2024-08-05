from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import logging

def train_logistic_regression(x_train, y_train):
    """Train a Logistic Regression model."""
    try:
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        model = LogisticRegression().fit(x_train_scaled, y_train)
        logging.info("Logistic Regression model training completed successfully.")
        return model
    except Exception as e:
        logging.error(f"Logistic Regression model training failed: {e}")
        raise

def train_random_forest(x_train, y_train):
    """Train the Random Forest model."""
    try:
        rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=123)
        rf.fit(x_train, y_train)
        logging.info("Random Forest model training completed successfully.")
        return rf
    except Exception as e:
        logging.error(f"Random Forest model training failed: {e}")
        raise
