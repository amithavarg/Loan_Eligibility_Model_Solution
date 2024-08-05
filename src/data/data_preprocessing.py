import pandas as pd
import logging

def preprocess_data(df):
    """Preprocess the data: handle missing values and encode categorical variables."""
    try:
        # Handle missing values
        df['Gender'].fillna('Male', inplace=True)
        df['Married'].fillna(df['Married'].mode()[0], inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
        df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
        
        # Drop 'Loan_ID' variable
        df = df.drop('Loan_ID', axis=1)
        
        # Create dummy variables for categorical variables
        df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], drop_first=True)
        
        logging.info("Data preprocessing completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")
        raise

def split_data(df):
    """Split the data into training and testing sets."""
    try:
        x = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
        logging.info("Data splitting completed successfully.")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Data splitting failed: {e}")
        raise
