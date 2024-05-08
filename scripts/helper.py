import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean the DataFrame by handling missing values and outliers."""
    # Fill missing values with 'No' for categorical data (assuming missing as 'not provided')
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna('No', inplace=True)
    
    # Impute numeric columns with the mean
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    
    return df

def preprocess_data(df):
    """Preprocess the DataFrame: encode categorical variables and scale numerical ones."""
    # Define categorical and numeric features
    categorical_feats = df.select_dtypes(include=['object']).columns
    numeric_feats = df.select_dtypes(include=['float64', 'int64']).columns

    # Define transformations
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformations into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_feats),
            ('cat', categorical_transformer, categorical_feats)
        ])

    # Apply transformations
    df_preprocessed = preprocessor.fit_transform(df)
    return df_preprocessed, preprocessor

def prepare_data(filepath):
    """Load, clean, and preprocess data from a CSV file."""
    df = load_data(filepath)
    df_cleaned = clean_data(df)
    df_preprocessed, preprocessor = preprocess_data(df_cleaned)
    return df_preprocessed, preprocessor

# The module can be used to prepare data like this:
# df_processed, preprocessor = prepare_data('path_to_your_data.csv')
