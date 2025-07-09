#Best Model so far

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers


#reproducability 
#np.random.seed(42)
#random.seed(42)
#tf.random.set_seed(42)

# Load the Excel file
file_path = '/Users/alexd/Documents/Davidson Research/WholeBrainBetas.xlsx'

# Step 1: Read the outcome sheet
outcome_df = pd.read_excel(file_path, sheet_name='outcomes')

# Step 2: Extract unique IDs (though you may not need this if you are using the outcome_df directly)
relevant_ids = outcome_df['SID'].unique()  # Extract unique IDs

# Read all sheets into a dictionary of DataFrames, except the outcome sheet
sheets_dict = pd.read_excel(file_path, sheet_name=None)

# Step 3: Merge each sheet with the outcome DataFrame
merged_df = outcome_df.copy()  # Start with the outcome DataFrame

for sheet_name, df in sheets_dict.items():
    if sheet_name not in ['outcomes', 'behavioral', 'demographics']:
        # Remove duplicates based on 'SID' in the current sheet
        df = df.drop_duplicates(subset=['SID'])
        
        # Merge the current sheet with the outcome DataFrame based on 'SID'
        merged_df = merged_df.merge(df, on='SID', how='left', suffixes=('', f'_{sheet_name}'))

# Display the final merged DataFrame
print(merged_df)

# Extract features and target
X = merged_df.drop(columns=['Imp20PercentBPRS', 'SID'])
y = merged_df['Imp20PercentBPRS']

# Separate categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

# Preprocessor: scaling for numerical and one-hot encoding for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# PCA to reduce dimensionality
pca = PCA(n_components=50)  # Adjust components based on your needs

# Initialize Leave-One-Out Cross-Validation
loo = LeaveOneOut()

# Arrays to hold results
y_true = []
y_pred = []

# Define the model outside of the loop
model = keras.Sequential([
    layers.Input(shape=(50,)),  # Use Input layer to specify input shape
    layers.Dense(64),
    layers.LeakyReLU(negative_slope=0.01),
    layers.Dropout(0.3),
    layers.Dense(32),
    layers.LeakyReLU(negative_slope=0.01),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the Model once
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Iterate through each LOO split
for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Preprocessing: Scaling and PCA
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Apply PCA after scaling/encoding
    X_train_pca = pca.fit_transform(X_train_processed)
    X_test_pca = pca.transform(X_test_processed)

    # Train the Model (with reduced epochs for LOO CV to improve speed)
    model.fit(X_train_pca, y_train, epochs=10, batch_size=32, verbose=0)  # Reduced epochs for speed
    
    # Make prediction for this test case
    y_pred_prob = model.predict(X_test_pca)
    y_pred.append((y_pred_prob > 0.5).astype("int32")[0][0])
    y_true.append(y_test.values[0])

# Evaluate LOO results
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred))



#best results 
# Confusion Matrix:
# [[31  4]
#  [ 4 43]]

# Classification Report:
#               precision    recall  f1-score   support

#            0       0.89      0.89      0.89        35
#            1       0.91      0.91      0.91        47

#     accuracy                           0.90        82
#    macro avg       0.90      0.90      0.90        82
# weighted avg       0.90      0.90      0.90        82


