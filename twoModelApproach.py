import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold
from collections import Counter

from sklearn.model_selection import StratifiedKFold


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def load_data(file_path):
    """Load datasets from the specified Excel file."""
    outcome_df = pd.read_excel(file_path, sheet_name='outcomes')
    scanner_ids = pd.read_excel(file_path, sheet_name='scannerid')
    subset_scanner_ids = scanner_ids[scanner_ids['EP1or2'] == 1]
    usable_outcomes = outcome_df[outcome_df['SID'].isin(subset_scanner_ids['SID'])]
    fmrifeatures_df = pd.read_excel(file_path, sheet_name='rawfmrifeatures') 
    fmrifeatures_df = fmrifeatures_df[fmrifeatures_df['SID'].isin(subset_scanner_ids['SID'])]

    clinfeatues_df = pd.read_excel(file_path, sheet_name = 'clinfeatures')
    clinfeatues_df = clinfeatues_df[clinfeatues_df['SID'].isin(subset_scanner_ids['SID'])]

    usable_outcomes = usable_outcomes.merge(clinfeatues_df,on='SID', how='inner')

    return usable_outcomes, fmrifeatures_df

def load_data2(file_path):
    """Load datasets from the specified Excel file."""
    outcome_df = pd.read_excel(file_path, sheet_name='outcomes')
    scanner_ids = pd.read_excel(file_path, sheet_name='scannerid')
    subset_scanner_ids = scanner_ids[scanner_ids['EP1or2'] == 2]
    usable_outcomes = outcome_df[outcome_df['SID'].isin(subset_scanner_ids['SID'])]
    fmrifeatures_df = pd.read_excel(file_path, sheet_name='rawfmrifeatures') 
    fmrifeatures_df = fmrifeatures_df[fmrifeatures_df['SID'].isin(subset_scanner_ids['SID'])]

    clinfeatues_df = pd.read_excel(file_path, sheet_name = 'clinfeatures')
    clinfeatues_df = clinfeatues_df[clinfeatues_df['SID'].isin(subset_scanner_ids['SID'])]

    usable_outcomes = usable_outcomes.merge(clinfeatues_df,on='SID', how='inner')

    return usable_outcomes, fmrifeatures_df


def merge_demographics_data(usable_outcomes, cueB_minus_cueA, file_path):
    """merging the demographics sheet"""
    demographic_df = pd.read_excel(file_path, sheet_name='demographics')
    demographic_df = demographic_df[demographic_df['SID'].isin(usable_outcomes['SID'])]
    merged_demographic_df_with_SID = usable_outcomes.merge(demographic_df, on='SID', how='inner')
    merged_demographic_df_with_SID = usable_outcomes.merge(cueB_minus_cueA, on='SID', how='inner')
    return merged_demographic_df_with_SID.drop(columns=['Chg_BPRS','SID'])


def prepare_data(merged_df):
    """Prepare features and labels from the merged DataFrame."""
    X = merged_df.drop(columns=['Imp20PercentBPRS']).values
    y = merged_df['Imp20PercentBPRS'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def merge_data(usable_outcomes, cueB_minus_cueA):
    """Merge usable outcomes with CueB - CueA differences and remove identical rows."""
    merged_df_with_SID = usable_outcomes.merge(cueB_minus_cueA, on='SID', how='inner')
    merged_df_with_SID = merged_df_with_SID.drop_duplicates()
    return merged_df_with_SID.drop(columns=['Chg_BPRS', 'SID'])

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



def process_and_prepare_data(file_path, sid):
    """
    Function to process and prepare data for modeling.
    Steps:
    1. Load usable outcomes and fMRI features
    2. Calculate cue differences
    3. Merge data
    4. Merge demographic data
    5. Prepare final features (X) and labels (y)
    
    Parameters:
    - file_path: path to the dataset
    - sid: an identifier to decide which dataset to use (1 or 2)
    
    Returns:
    - X_demos: Feature matrix with demographics
    - y_demos: Target labels
    """
    # Load the data based on the sid (identifier)
    if sid == 2:
        usable_outcomes, fmrifeatures_df = load_data2(file_path)
    else:
        usable_outcomes, fmrifeatures_df = load_data(file_path)

    # Calculate the cue differences
    cueB_minus_cueA = calculate_cue_differences(fmrifeatures_df)
    
    # Merge the usable outcomes with the cue differences
    merged_df = merge_data(usable_outcomes, cueB_minus_cueA)
    
    # Merge the demographic data
    merged_df_with_demos = merge_demographics_data(usable_outcomes, cueB_minus_cueA, file_path)
    
    # Prepare the final features (X) and labels (y)
    X, y = prepare_data(merged_df_with_demos)
    
    return X, y



#=====================================Data Preprocessing Finished ===========================================================
def create_confidence_model(input_shape, learning_rate=0.001, dropout_rate=0.3):
    inputs = layers.Input(shape=(input_shape,))
    
    x = layers.Dense(128)(inputs)
    x = layers.LeakyReLU(negative_slope=0.01)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Two outputs
    prediction_output = layers.Dense(1, activation='sigmoid', name='prediction')(x)
    confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[prediction_output, confidence_output])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={'prediction': 'binary_crossentropy', 'confidence': 'mse'},
        loss_weights={'prediction': 1.0, 'confidence': 0.1},  # Adjust this based on experimentation
        metrics={'prediction': 'accuracy'}
    )
    
    return model


def create_model(input_shape, learning_rate=0.001, dropout_rate=0.3):
    model = Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout_rate),

        layers.Dense(64),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout_rate),

        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def choose_model_with_higher_confidence(model1, model2, input_sample):
    input_sample = input_sample.reshape(1, -1)
    
    # Predict both models
    pred1, conf1 = model1.predict(input_sample)
    pred2, conf2 = model2.predict(input_sample)

    print(f"confidence 1: {conf1[0][0]}")
    print(f"confidence 2: {conf2[0][0]}")

    # Compare confidence
    if conf1[0][0] > conf2[0][0]:
        print("Using EP1 model")
        return pred1[0][0], conf1[0][0]
    else:
        print("Using EP2 model")
        return pred2[0][0], conf2[0][0]

def train_model(X, y):
    # Create model
    model = create_confidence_model(X.shape[1])

    # Initialize an array for the confidence labels (0 for incorrect, 1 for correct)
    predicted_class = np.zeros_like(y)  # This will hold the confidence targets (0 or 1)

    # Make predictions on the data to calculate confidence
    predictions = model.predict(X)

    # Determine predicted classes (assuming a binary classification task)
    predicted_class = (predictions[0] > 0.5).astype(int)  # Assuming the first output is the prediction logits

    # Set confidence labels (1 for correct, 0 for incorrect)
    confidence_labels = (predicted_class == y).astype(float)

    # Now, train the model with both prediction and confidence as targets
    model.fit(
        X, 
        {'prediction': y, 'confidence': confidence_labels},  # Use correct confidence labels
        epochs=20, 
        batch_size=32, 
        verbose=0
    )
    
    return model


def two_model_approach(file_path):
    ep1 = 1
    ep2 = 2
    # Load training data for both ep1 and ep2
    X_train_ep1, y_train_ep1 = process_and_prepare_data(file_path, ep1)
    
    # Split ep2 data into training and testing sets
    X_train_ep2, y_train_ep2 = process_and_prepare_data(file_path, ep2)
    X_train_ep2, X_test_ep2, y_train_ep2, y_test_ep2 = train_test_split(X_train_ep2, y_train_ep2, test_size=0.2, random_state=42)
    
    # Train model for ep1 on all its data
    model_ep1 = train_model(X_train_ep1, y_train_ep1)

    # Train model for ep2 on its training data
    model_ep2 = train_model(X_train_ep2, y_train_ep2)

    # Test the models on ep2 test data
    chosen_model_preds = []
    for i in range(X_test_ep2.shape[0]):
        sample = X_test_ep2[i]
        pred, _ = choose_model_with_higher_confidence(model_ep1, model_ep2, sample)
        chosen_model_preds.append(pred)
    
    # Calculate accuracy on the test data
    accuracy = accuracy_score(y_test_ep2, np.round(chosen_model_preds))
    print(f"Accuracy on ep2 test data: {accuracy}")




def run_kfold_cv_with_confidence(file_path, n_splits=10):
    # Load data for ep1 and ep2
    ep1 = 1
    ep2 = 2
    X_train_ep1, y_train_ep1 = process_and_prepare_data(file_path, ep1)
    X_train_ep2, y_train_ep2 = process_and_prepare_data(file_path, ep2)
    
    # Initialize StratifiedKFold for 10-fold cross-validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Variables to store results
    ep2_fold_accuracies = []
    ep2_fold_aucs = []  # To store AUC scores

    # Iterate through the folds for ep1 and ep2 data
    for train_index, test_index in kfold.split(X_train_ep2, y_train_ep2):
        X_train_fold, X_test_fold = X_train_ep2[train_index], X_train_ep2[test_index]
        y_train_fold, y_test_fold = y_train_ep2[train_index], y_train_ep2[test_index]

        # Train model on ep1 (using all data from ep1)
        model_ep1 = train_model(X_train_ep1, y_train_ep1)
        
        # Train model on ep2 fold (using the current training portion of ep2)
        model_ep2 = train_model(X_train_fold, y_train_fold)
        
        # Store the probabilities instead of rounded predictions
        chosen_model_probs = []

        for i in range(X_test_fold.shape[0]):
            sample = X_test_fold[i]
            pred, _ = choose_model_with_higher_confidence(model_ep1, model_ep2, sample)
            chosen_model_probs.append(pred)  # Save the predicted probability
        
        # Calculate accuracy for the current fold (optional)
        fold_accuracy = accuracy_score(y_test_fold, np.round(chosen_model_probs))
        ep2_fold_accuracies.append(fold_accuracy)
        
        # Calculate AUC for the current fold
        auc = roc_auc_score(y_test_fold, chosen_model_probs)
        ep2_fold_aucs.append(auc)

    # Calculate the average accuracy and AUC over all folds
    avg_accuracy_ep2 = np.mean(ep2_fold_accuracies)
    avg_auc_ep2 = np.mean(ep2_fold_aucs)
    
    print(f"Average accuracy on ep2 test data using 10-fold CV: {avg_accuracy_ep2}")
    print(f"Average AUC on ep2 test data using 10-fold CV: {avg_auc_ep2}")

#=====================================Model END===========================================================


file_path = "/Users/alexd/Documents/Davidson Research/WholeBrainBetas.xlsx"
run_kfold_cv_with_confidence(file_path)
#two_model_approach(file_path)
