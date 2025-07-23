import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#Load the data
data_frame = pd.read_csv('Competency_Test_Contracts_20250721.csv')

#1 Data Cleaning
# 1.1 Fill null values in 'Contract Type', 'Department' and dates fields 
# Identify rows with missing values
missing_contract_type = data_frame['Contract Type'].isna()
missing_department = data_frame['Department'].isna()

# Extract features from text columns for prediction
feature_cols = ['Purchase Order (Contract) Number', 'Purchase Order Description', 'Specification Number']
features = pd.DataFrame(index=data_frame.index)

for col in feature_cols:
    # Convert column to string and handle NaN
    str_col = data_frame[col].fillna('').astype(str)
    
    # Extract text features
    features[f'{col}_length'] = str_col.str.len()
    features[f'{col}_word_count'] = str_col.str.count(r'\s+') + 1
    features[f'{col}_digit_count'] = str_col.str.count(r'[0-9]')
    features[f'{col}_alpha_count'] = str_col.str.count(r'[A-Za-z]')
    
    # Avoid division by zero
    length_nonzero = features[f'{col}_length'].replace(0, 1)
    features[f'{col}_digit_ratio'] = features[f'{col}_digit_count'] / length_nonzero
    features[f'{col}_alpha_ratio'] = features[f'{col}_alpha_count'] / length_nonzero

# Function to impute missing values using KNN
def knn_impute(df, features_df, column_name, missing_mask):
    if not missing_mask.any():
        return  # No missing values to impute
    
    # Get training and imputation data
    X_train = features_df.loc[~missing_mask].values
    y_train = df.loc[~missing_mask, column_name].values
    X_impute = features_df.loc[missing_mask].values
    
    # Calculate appropriate number of neighbors
    n_samples = len(y_train)
    n_neighbors = min(int(np.sqrt(n_samples)), 10)
    n_neighbors = max(1, min(n_neighbors, n_samples))
    
    # Fit KNN model and predict
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn_model.fit(X_train, y_train)
    df.loc[missing_mask, column_name] = knn_model.predict(X_impute)

# Impute missing values for target columns
knn_impute(data_frame, features, 'Contract Type', missing_contract_type)
knn_impute(data_frame, features, 'Department', missing_department)
    
# Keeping null values in date columns since they're used in later in Data Transformations.

# 1.2 Convert date fields to standard format
date_columns = ['Start Date', 'End Date', 'Approval Date']
for column in date_columns:
    data_frame[column] = pd.to_datetime(data_frame[column], errors='coerce')

# 2 Data Transformation 
# Create Contract Duration column
data_frame['Contract Duration (days)'] = None
date_mask = data_frame['Start Date'].notnull() & data_frame['End Date'].notnull()
data_frame.loc[date_mask, 'Contract Duration (days)'] = (
    pd.to_datetime(data_frame.loc[date_mask, 'End Date']) - 
    pd.to_datetime(data_frame.loc[date_mask, 'Start Date'])
).dt.days

# Create Is Blanket Contract column
data_frame['Is Blanket Contract'] = data_frame['End Date'].notnull()

# Create Legacy Record column
data_frame['Legacy Record'] = data_frame['Purchase Order (Contract) Number'].astype(str).str.match(r'^[A-Za-z]').fillna(False)

# Create Has Negative Modification column
data_frame['Has Negative Modification'] = data_frame['Award Amount'] < 0

# Save the processed data

data_frame.to_csv('Competency_Test_Contracts_20250721_rf_predicted.csv', index=False)