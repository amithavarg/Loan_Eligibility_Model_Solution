from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from src.utils.utilities import save_figure

def evaluate_model(model, x_train, x_test, y_train, y_test, model_name):
    """Evaluate the model with accuracy score and confusion matrix."""
    try:
        # Predict on test set
        y_pred = model.predict(x_test)
        
        # Accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"{model_name} Accuracy: {accuracy}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Denied', 'Approved'], yticklabels=['Denied', 'Approved'])
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        save_figure(plt.gcf(), f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.show()

        logging.info(f"{model_name} evaluation completed successfully.")
    except Exception as e:
        logging.error(f"{model_name} evaluation failed: {e}")
        raise
