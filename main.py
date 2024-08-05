from src.data.data_loader import load_data
from src.data.data_preprocessing import preprocess_data, split_data
from src.data.data_visualization import visualize_data
from src.model.model_training import train_logistic_regression, train_random_forest
from src.model.model_evaluation import evaluate_model
from src.utils.utilities import configure_logging

def main():
    logger = configure_logging()

    try:
        # Load the data
        df = load_data('data/credit.csv')
        logger.info("Data loaded successfully.")

        # Preprocess the data
        df = preprocess_data(df)
        logger.info("Data preprocessed successfully.")

        # Visualize the data
        visualize_data(df)
        logger.info("Data visualized successfully.")

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = split_data(df)
        logger.info("Data split into training and testing sets successfully.")

        # Train Logistic Regression model
        lr_model = train_logistic_regression(x_train, y_train)
        logger.info("Logistic Regression model trained successfully.")
        evaluate_model(lr_model, x_train, x_test, y_train, y_test, model_name='Logistic Regression')

        # Train Random Forest model
        rf_model = train_random_forest(x_train, y_train)
        logger.info("Random Forest model trained successfully.")
        evaluate_model(rf_model, x_train, x_test, y_train, y_test, model_name='Random Forest')

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
