import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss
from sklearn.utils import shuffle



def mixup(X, y, alpha=0.2, n_samples=None):
    # Determine the number of samples to generate
    if n_samples is None:
        n_samples = X.shape[0]
    
    # Initialize arrays for mixed features and labels
    mixed_X = []
    mixed_y = []

    for _ in range(n_samples):
        # Randomly sample two indices
        idx1, idx2 = np.random.choice(X.shape[0], 2, replace=False)

        # Extract the corresponding data
        x1, x2 = X[idx1], X[idx2]
        y1, y2 = y[idx1], y[idx2]

        # Sample a lambda value from the beta distribution
        lambda_value = np.random.beta(alpha, alpha)

        # Generate mixed features and labels
        mixed_X.append(lambda_value * x1 + (1 - lambda_value) * x2)
        mixed_y.append(lambda_value * y1 + (1 - lambda_value) * y2)

    # Convert to numpy arrays
    mixed_X = np.array(mixed_X)
    mixed_y = np.array(mixed_y)

    return mixed_X, mixed_y


def calculate_cue_differences(fmrifeatures_df):
    """Calculate CueB - CueA differences and return as a DataFrame."""
    new_columns = {'SID': fmrifeatures_df['SID']}
    for col in fmrifeatures_df.columns:
        if col.startswith('CueA_'):
            variable_part = col[5:]
            cueb_col = f'CueB_{variable_part}'
            if cueb_col in fmrifeatures_df.columns:
                new_col_name = f'CueBMinusA_{variable_part}'
                new_columns[new_col_name] = fmrifeatures_df[cueb_col] - fmrifeatures_df[col]
    return pd.DataFrame(new_columns)

def merge_data(usable_outcomes, cueB_minus_cueA):
    """Merge usable outcomes with CueB - CueA differences and remove identical rows."""
    merged_df_with_SID = usable_outcomes.merge(cueB_minus_cueA, on='SID', how='inner')
    merged_df_with_SID = merged_df_with_SID.drop_duplicates()
    return merged_df_with_SID.drop(columns=['Chg_BPRS', 'SID'])

def prepare_data_for_logistic_regression(data):
    """Prepares data for logistic regression by separating features and target variable."""
    X = data.drop(columns=['Imp20PercentBPRS'])  # Drop the target column
    y = data['Imp20PercentBPRS']  # Use the target column for labels
    return X, y


def logistic_regression_model(file_path):

    #=============================================Data preprocessing Starts=============================================
    
    latent_dim = 64  # Latent dimension size
    epochs = 20  # Number of epochs
    outcome_df = pd.read_excel(file_path, sheet_name='outcomes')
    scanner_ids = pd.read_excel(file_path, sheet_name='scannerid')
    fmrifeatures_df = pd.read_excel(file_path, sheet_name='rawfmrifeatures')
    demographics_df = pd.read_excel(file_path, sheet_name='demographics')
    #reading in the sheets needed 

    subset_scanner_ids_EP1 = scanner_ids[scanner_ids['EP1or2'] == 1]
    subset_scanner_ids_EP2 = scanner_ids[scanner_ids['EP1or2'] == 2]
    #dividing the scanner ids into different scanners ep1 or ep2


    usable_outcomes_EP1 = outcome_df[outcome_df['SID'].isin(subset_scanner_ids_EP1['SID'])]
    usable_outcomes_EP2 = outcome_df[outcome_df['SID'].isin(subset_scanner_ids_EP2['SID'])]  
    #dividing outcomes into EP1 or EP2

    fmrifeatures_df_EP1 = fmrifeatures_df[fmrifeatures_df['SID'].isin(subset_scanner_ids_EP1['SID'])]
    fmrifeatures_df_EP2 = fmrifeatures_df[fmrifeatures_df['SID'].isin(subset_scanner_ids_EP2['SID'])]
    #dividing features into EP1 or EP2

    demographics_df_EP1 = demographics_df[demographics_df['SID'].isin(subset_scanner_ids_EP1['SID'])]
    demographics_df_EP2 = demographics_df[demographics_df['SID'].isin(subset_scanner_ids_EP2['SID'])]
    
    demographics_df_EP1 = demographics_df_EP1.drop(columns='Dx')
    demographics_df_EP2 = demographics_df_EP2.drop(columns='Dx')

    cueB_minus_cueA_EP1 = calculate_cue_differences(fmrifeatures_df_EP1)
    cueB_minus_cueA_EP2 = calculate_cue_differences(fmrifeatures_df_EP2)
    #calculdaing cue differences

    #combining demographics and cueDiffs
    cueB_minus_cueA_EP1 = cueB_minus_cueA_EP1.merge(demographics_df_EP1, on='SID', how='inner')
    cueB_minus_cueA_EP2 = cueB_minus_cueA_EP2.merge(demographics_df_EP2, on='SID', how='inner')

    ep1_data = merge_data(usable_outcomes_EP1, cueB_minus_cueA_EP1)
    ep2_data = merge_data(usable_outcomes_EP2, cueB_minus_cueA_EP2)

    scaler = MinMaxScaler()
    # Apply MinMax scaling to the data
    ep1_data_normalized = scaler.fit_transform(ep1_data)
    ep2_data_normalized = scaler.transform(ep2_data)  # Use transform on EP2 to avoid data leakage
    ep1_data = pd.DataFrame(ep1_data_normalized, columns=ep1_data.columns)
    ep2_data = pd.DataFrame(ep2_data_normalized, columns=ep2_data.columns)


#=============================================Data preprocessing ends=============================================
    # Prepare data for EP1 and EP2
    X_ep1, y_ep1 = prepare_data_for_logistic_regression(ep1_data)
    X_ep2, y_ep2 = prepare_data_for_logistic_regression(ep2_data)

    # Split the data into training and testing sets



    X_train_ep1, X_test_ep1, y_train_ep1, y_test_ep1 = train_test_split(X_ep1, y_ep1, test_size=0.2, random_state=42)
    X_train_ep2, X_test_ep2, y_train_ep2, y_test_ep2 = train_test_split(X_ep2, y_ep2, test_size=0.2, random_state=42)

    #X_train_ep1, y_train_ep1 = shuffle(X_train_ep1, y_train_ep1, random_state=42)
    #X_train_ep2, y_train_ep2 = shuffle(X_train_ep2, y_train_ep2, random_state=42)
    
    # Initialize the logistic regression model
    logreg_model_ep1 = LogisticRegression(max_iter=1000)
    logreg_model_ep2 = LogisticRegression(max_iter=10)

    # Train the model on EP1 data
    logreg_model_ep1.fit(X_train_ep1, y_train_ep1)

    # Train the model on EP2 data
    logreg_model_ep2.fit(X_train_ep2, y_train_ep2)


    # Predictions on EP1 test data
    y_pred_ep1 = logreg_model_ep1.predict(X_test_ep1)
    y_pred_prob_ep1 = logreg_model_ep1.predict_proba(X_test_ep1)[:, 1]  # For AUC calculation

    # Predictions on EP2 test data
    y_pred_ep2 = logreg_model_ep2.predict(X_test_ep2)
    y_pred_prob_ep2 = logreg_model_ep2.predict_proba(X_test_ep2)[:, 1]  # For AUC calculation


    # Predictions for EP1
    y_pred_prob_ep1 = logreg_model_ep1.predict_proba(X_test_ep1)[:, 1]  # Probability predictions
    print(f"Log Loss for EP1: {log_loss(y_test_ep1, y_pred_prob_ep1)}")

    # Predictions for EP2
    y_pred_prob_ep2 = logreg_model_ep2.predict_proba(X_test_ep2)[:, 1]  # Probability predictions
    print(f"Log Loss for EP2: {log_loss(y_test_ep2, y_pred_prob_ep2)}")


    # Evaluate EP1 model
    print("EP1 Results")
    print(f"Accuracy: {accuracy_score(y_test_ep1, y_pred_ep1)}")
    print(f"Classification Report: \n{classification_report(y_test_ep1, y_pred_ep1)}")
    print(f"Confusion Matrix: \n{confusion_matrix(y_test_ep1, y_pred_ep1)}")
    print(f"ROC AUC Score: {roc_auc_score(y_test_ep1, y_pred_prob_ep1)}")

    # Evaluate EP2 model
    print("\nEP2 Results")
    print(f"Accuracy: {accuracy_score(y_test_ep2, y_pred_ep2)}")
    print(f"Classification Report: \n{classification_report(y_test_ep2, y_pred_ep2)}")
    print(f"Confusion Matrix: \n{confusion_matrix(y_test_ep2, y_pred_ep2)}")
    print(f"ROC AUC Score: {roc_auc_score(y_test_ep2, y_pred_prob_ep2)}")


file_path = '/Users/alexd/Documents/Davidson Research/WholeBrainBetas.xlsx'
logistic_regression_model(file_path)
