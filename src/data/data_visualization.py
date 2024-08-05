import seaborn as sns
import matplotlib.pyplot as plt
import logging
from src.utils.utilities import save_figure

def visualize_data(df):
    """Visualize the data: missing values, distributions, etc."""
    try:
        # Missing values
        missing_values = df.isnull().sum()
        logging.info(f"Missing values:\n{missing_values}")
        
        # Distribution of Loan Amount
        plt.figure(figsize=(10, 6))
        sns.histplot(df['LoanAmount'], kde=True)
        plt.title('Distribution of Loan Amount')
        save_figure(plt.gcf(), 'loan_amount_distribution.png')
        plt.show()
        
        # Loan Approved vs. Loan Denied
        plt.figure(figsize=(10, 6))
        df['Loan_Status'].value_counts().plot(kind='bar')
        plt.title('Loan Approved vs. Loan Denied')
        save_figure(plt.gcf(), 'loan_status_vs_denied.png')
        plt.show()

        logging.info("Data visualization completed successfully.")
    except Exception as e:
        logging.error(f"Data visualization failed: {e}")
        raise
