"""import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

def create_model():
    
    Creates and trains the Extra Trees model with the same preprocessing pipeline
    as used in the original analysis.
    
    # Create synthetic training data based on the analysis patterns
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data with realistic distributions
    data = {
        'age': np.random.normal(45, 16, n_samples).clip(4, 90),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.75, 0.25]),
        'total_bilirubin': np.random.lognormal(0.5, 1.2, n_samples).clip(0.4, 75),
        'direct_bilirubin': np.random.lognormal(0.2, 1.0, n_samples).clip(0.1, 19.7),
        'alkaline_phosphotase': np.random.lognormal(5.5, 0.6, n_samples).clip(63, 2110),
        'alamine_aminotransferase': np.random.lognormal(3.5, 1.2, n_samples).clip(10, 2000),
        'aspartate_aminotransferase': np.random.lognormal(3.8, 1.3, n_samples).clip(10, 4929),
        'total_protiens': np.random.normal(6.5, 1.1, n_samples).clip(2.7, 9.6),
        'albumin': np.random.normal(3.1, 0.8, n_samples).clip(0.9, 5.5),
        'albumin_and_globulin_ratio': np.random.normal(0.95, 0.32, n_samples).clip(0.3, 2.8)
    }
    
    # Create target variable with realistic correlations
    df = pd.DataFrame(data)
    
    # Create target based on feature combinations (mimicking liver disease patterns)
    risk_score = (
        (df['total_bilirubin'] > 2.0).astype(int) * 2 +
        (df['direct_bilirubin'] > 1.0).astype(int) * 2 +
        (df['alkaline_phosphotase'] > 300).astype(int) +
        (df['alamine_aminotransferase'] > 60).astype(int) +
        (df['aspartate_aminotransferase'] > 80).astype(int) +
        (df['albumin'] < 3.0).astype(int) +
        (df['age'] > 55).astype(int)
    )
    
    # Create target with some randomness
    df['dataset'] = np.where(
        risk_score >= 3, 
        np.random.choice([1, 2], n_samples, p=[0.3, 0.7]),
        np.random.choice([1, 2], n_samples, p=[0.8, 0.2])
    )
    
    # Apply the same preprocessing as in the original analysis
    df = preprocess_training_data(df)
    
    # Split features and target
    X = df.drop('dataset', axis=1)
    y = df['dataset']
    
    # Balance the data (upsample minority class)
    df_combined = pd.concat([X, y], axis=1)
    minority = df_combined[df_combined.dataset == 2]
    majority = df_combined[df_combined.dataset == 1]
    
    if len(minority) < len(majority):
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        df_balanced = pd.concat([minority_upsampled, majority])
    else:
        df_balanced = df_combined
    
    X_balanced = df_balanced.drop('dataset', axis=1)
    y_balanced = df_balanced['dataset']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.25, random_state=123
    )
    
    # Train Extra Trees model with optimized parameters
    model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Create preprocessing objects fitted on training data
    le = LabelEncoder()
    le.fit(['Female', 'Male'])
    
    # Fit robust scaler on training data
    rs = RobustScaler()
    numerical_features = ['age', 'gender', 'total_bilirubin', 'alkaline_phosphotase', 
                         'alamine_aminotransferase', 'albumin_and_globulin_ratio']
    rs.fit(X_train[numerical_features])
    
    return {
        'model': model,
        'label_encoder': le,
        'robust_scaler': rs,
        'feature_columns': X_train.columns.tolist()
    }

def preprocess_training_data(df):
    
    Apply the same preprocessing steps as in the original analysis for training data.
    
    # Handle missing values (fill with mean)
    df['albumin_and_globulin_ratio'].fillna(df['albumin_and_globulin_ratio'].mean(), inplace=True)
    
    # Drop highly correlated features (as done in original analysis)
    features_to_drop = ['direct_bilirubin', 'aspartate_aminotransferase', 'total_protiens', 'albumin']
    df = df.drop([col for col in features_to_drop if col in df.columns], axis=1)
    
    # Apply log1p transformation to skewed features
    skewed_cols = ['albumin_and_globulin_ratio', 'total_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase']
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    
    # Encode gender
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    
    # Scale numerical features
    rs = RobustScaler()
    numerical_features = ['age', 'gender', 'total_bilirubin', 'alkaline_phosphotase', 
                         'alamine_aminotransferase', 'albumin_and_globulin_ratio']
    
    for col in numerical_features:
        if col in df.columns:
            df[col] = rs.fit_transform(df[col].values.reshape(-1, 1)).flatten()
    
    return df

def preprocess_input(input_df, model_data):
    
    Preprocess user input using the same pipeline as training data.
    
    # Make a copy to avoid modifying original
    df = input_df.copy()
    
    # Handle missing values (shouldn't happen with web input, but just in case)
    if df['albumin_and_globulin_ratio'].isna().any():
        df['albumin_and_globulin_ratio'].fillna(0.947, inplace=True)  # Mean from original data
    
    # Drop highly correlated features (same as training)
    features_to_drop = ['direct_bilirubin', 'aspartate_aminotransferase', 'total_protiens', 'albumin']
    df = df.drop([col for col in features_to_drop if col in df.columns], axis=1)
    
    # Apply log1p transformation to skewed features
    skewed_cols = ['albumin_and_globulin_ratio', 'total_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase']
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    
    # Encode gender using fitted encoder
    df['gender'] = model_data['label_encoder'].transform(df['gender'])
    
    # Scale numerical features using fitted scaler
    numerical_features = ['age', 'gender', 'total_bilirubin', 'alkaline_phosphotase', 
                         'alamine_aminotransferase', 'albumin_and_globulin_ratio']
    
    df[numerical_features] = model_data['robust_scaler'].transform(df[numerical_features])
    
    # Ensure column order matches training data
    df = df[model_data['feature_columns']]
    
    return df"""


import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings
import os
warnings.filterwarnings('ignore')

def create_model():
    """
    Creates and trains the Extra Trees model using the real Indian Liver Patient dataset
    with the same preprocessing pipeline as used in the original analysis.
    """
    # Check if dataset exists
    if not os.path.exists('indian_liver_patient.csv'):
        raise FileNotFoundError("Dataset file 'indian_liver_patient.csv' not found")
    
    # Load the real dataset
    df = pd.read_csv('indian_liver_patient.csv')
    
    # Convert column names to lowercase as in original analysis
    df.columns = df.columns.str.lower()
    
    # Handle missing values (fill with mean)
    missing_count = df['albumin_and_globulin_ratio'].isnull().sum()
    if missing_count > 0:
        df['albumin_and_globulin_ratio'].fillna(df['albumin_and_globulin_ratio'].mean(), inplace=True)
    
    # Apply the same preprocessing as in the original analysis
    df = preprocess_training_data(df)
    
    # Split features and target
    X = df.drop('dataset', axis=1)
    y = df['dataset']
    
    # Balance the data (upsample minority class)
    df_combined = pd.concat([X, y.to_frame()], axis=1)
    minority = df_combined[df_combined['dataset'] == 2]
    majority = df_combined[df_combined['dataset'] == 1]
    
    if len(minority) < len(majority):
        minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
        df_balanced = pd.concat([minority_upsampled, majority], ignore_index=True)
    else:
        df_balanced = df_combined
    
    X_balanced = df_balanced.drop('dataset', axis=1)
    y_balanced = df_balanced['dataset']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.25, random_state=123
    )
    
    # Train Extra Trees model with optimized parameters from GridSearchCV
    # Best parameters: {'criterion': 'gini', 'max_leaf_nodes': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    model = ExtraTreesClassifier(
        n_estimators=200,
        criterion='gini',
        max_leaf_nodes=None,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Create preprocessing objects fitted on training data
    le = LabelEncoder()
    le.fit(['Female', 'Male'])
    
    # Fit robust scaler on training data
    rs = RobustScaler()
    numerical_features = ['age', 'gender', 'total_bilirubin', 'alkaline_phosphotase', 
                         'alamine_aminotransferase', 'albumin_and_globulin_ratio']
    
    # Create numerical feature subset for fitting
    X_train_numerical = X_train.loc[:, numerical_features]
    rs.fit(X_train_numerical)
    
    # Store feature column names
    feature_columns_list = X_train.columns.tolist()
    
    return {
        'model': model,
        'label_encoder': le,
        'robust_scaler': rs,
        'feature_columns': feature_columns_list
    }

def preprocess_training_data(df):
    """
    Apply the same preprocessing steps as in the original analysis for training data.
    """
    # Handle missing values (fill with mean)
    df['albumin_and_globulin_ratio'].fillna(df['albumin_and_globulin_ratio'].mean(), inplace=True)
    
    # Drop highly correlated features (as done in original analysis)
    features_to_drop = ['direct_bilirubin', 'aspartate_aminotransferase', 'total_protiens', 'albumin']
    df = df.drop([col for col in features_to_drop if col in df.columns], axis=1)
    
    # Apply log1p transformation to skewed features
    skewed_cols = ['albumin_and_globulin_ratio', 'total_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase']
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    
    # Encode gender
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    
    # Scale numerical features
    rs = RobustScaler()
    numerical_features = ['age', 'gender', 'total_bilirubin', 'alkaline_phosphotase', 
                         'alamine_aminotransferase', 'albumin_and_globulin_ratio']
    
    for col in numerical_features:
        if col in df.columns:
            df[col] = rs.fit_transform(df[col].values.reshape(-1, 1)).flatten()
    
    return df

def preprocess_input(input_df, model_data):
    """
    Preprocess user input using the same pipeline as training data.
    """
    # Make a copy to avoid modifying original
    df = input_df.copy()
    
    # Handle missing values (shouldn't happen with web input, but just in case)
    if df['albumin_and_globulin_ratio'].isna().any():
        df['albumin_and_globulin_ratio'].fillna(0.947, inplace=True)  # Mean from original data
    
    # Drop highly correlated features (same as training)
    features_to_drop = ['direct_bilirubin', 'aspartate_aminotransferase', 'total_protiens', 'albumin']
    df = df.drop([col for col in features_to_drop if col in df.columns], axis=1)
    
    # Apply log1p transformation to skewed features
    skewed_cols = ['albumin_and_globulin_ratio', 'total_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase']
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    
    # Encode gender using fitted encoder
    df['gender'] = model_data['label_encoder'].transform(df['gender'])
    
    # Scale numerical features using fitted scaler
    numerical_features = ['age', 'gender', 'total_bilirubin', 'alkaline_phosphotase', 
                         'alamine_aminotransferase', 'albumin_and_globulin_ratio']
    
    df_scaled = model_data['robust_scaler'].transform(df[numerical_features])
    df[numerical_features] = df_scaled
    
    # Ensure column order matches training data
    feature_columns = model_data['feature_columns']
    df = df[feature_columns]
    
    return df
