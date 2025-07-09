import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE  # Import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.decomposition import PCA


from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.metrics import Accuracy

from collections import Counter
from sklearn.model_selection import KFold


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

def without_subtracting_df(file_path):
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
    fmrifeatures_df_EP1 = fmrifeatures_df_EP1.drop_duplicates()
    fmrifeatures_df_EP2 = fmrifeatures_df_EP2.drop_duplicates()


    demographics_df_EP1 = demographics_df[demographics_df['SID'].isin(subset_scanner_ids_EP1['SID'])]
    demographics_df_EP2 = demographics_df[demographics_df['SID'].isin(subset_scanner_ids_EP2['SID'])]
    demographics_df_EP1 = demographics_df_EP1.drop(columns='Dx')
    demographics_df_EP2 = demographics_df_EP2.drop(columns='Dx')
    
    demographics_df_EP2 = demographics_df_EP2.drop_duplicates(subset='SID', keep='first')


    fmrifeatures_df_EP1 = fmrifeatures_df_EP1.merge(demographics_df_EP1, on='SID', how='inner')
    fmrifeatures_df_EP2 = fmrifeatures_df_EP2.merge(demographics_df_EP2, on='SID', how='inner')

    ep1_data = merge_data(usable_outcomes_EP1, fmrifeatures_df_EP1)
    ep2_data = merge_data(usable_outcomes_EP2, fmrifeatures_df_EP2)
    #print(ep1_data)
    #print(ep2_data)
    #getting rid of Chg_BPRS label and sid label 
    return ep1_data,ep2_data

def prepare_data(merged_df):
    """Prepare features and labels from the merged DataFrame."""
    X = merged_df.drop(columns=['Imp20PercentBPRS']).values
    y = merged_df['Imp20PercentBPRS'].values
    return X, y

def prepare_data_with_pca(merged_df, n_components=50):
    """Prepare features and labels from the merged DataFrame, including PCA for dimensionality reduction."""
    # Extract features (X) and labels (y)
    X = merged_df.drop(columns=['Imp20PercentBPRS']).values
    y = merged_df['Imp20PercentBPRS'].values
    
    # Standardize the features

    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=0.95)  # keeps enough components to explain 95% of variance
    X_pca = pca.fit_transform(X)
    
    return X_pca, y




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

def return_cueA_columns(fmrifeatures_df):
    """Return a DataFrame with 'SID' and all columns that start with 'CueA_'."""
    selected_cols = ['SID'] + [col for col in fmrifeatures_df.columns if col.startswith('CueA_')]
    cueA_df = fmrifeatures_df[selected_cols]
    return cueA_df

def return_cueB_columns(fmrifeatures_df):
    """Return a DataFrame with 'SID' and all columns that start with 'CueB_'."""
    selected_cols = ['SID'] + [col for col in fmrifeatures_df.columns if col.startswith('CueB_')]
    cueB_df = fmrifeatures_df[selected_cols]
    return cueB_df

def merge_data(usable_outcomes, cueB_minus_cueA):
    """Merge usable outcomes with CueB - CueA differences and remove identical rows."""
    merged_df_with_SID = usable_outcomes.merge(cueB_minus_cueA, on='SID', how='inner')
    merged_df_with_SID = merged_df_with_SID.drop_duplicates()
    return merged_df_with_SID.drop(columns=['Chg_BPRS', 'SID'])

def merge_demographics_data(usable_outcomes, cueB_minus_cueA, file_path):
    """merging the demographics sheet"""
    demographic_df = pd.read_excel(file_path, sheet_name='demographics')
    demographic_df = demographic_df[demographic_df['SID'].isin(usable_outcomes['SID'])]
    merged_demographic_df_with_SID = usable_outcomes.merge(demographic_df, on='SID', how='inner')
    merged_demographic_df_with_SID = usable_outcomes.merge(cueB_minus_cueA, on='SID', how='inner')
    return merged_demographic_df_with_SID.drop(columns=['Chg_BPRS','SID'])




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

def process_and_prepare_combined_data(file_path):
    """
    Function to process and prepare combined data from both load_data and load_data2.
    Adds an additional 'sid' column indicating whether the data came from load_data (1) or load_data2 (2).
    
    Steps:
    1. Load usable outcomes and fMRI features from both load_data and load_data2.
    2. Calculate cue differences.
    3. Merge data.
    4. Merge demographic data.
    5. Prepare final features (X) and labels (y).
    
    Parameters:
    - file_path: path to the dataset
    
    Returns:
    - X_combined: Combined feature matrix
    - y_combined: Combined target labels
    """
    # Load the data from both sources and add a 'sid' column to each
    usable_outcomes_1, fmrifeatures_df_1 = load_data(file_path)
    usable_outcomes_2, fmrifeatures_df_2 = load_data2(file_path)
    
    # Add a 'sid' column indicating the source for each dataset (1 for load_data, 2 for load_data2)

    
    # Merge the two datasets into one
    combined_fmrifeatures_df = pd.concat([fmrifeatures_df_1, fmrifeatures_df_2], axis=0, ignore_index=True)
    combined_usable_outcomes = pd.concat([usable_outcomes_1, usable_outcomes_2], axis=0, ignore_index=True)
    
    # Calculate the cue differences for both datasets
    cueB_minus_cueA_1 = calculate_cue_differences(fmrifeatures_df_1)
    cueB_minus_cueA_2 = calculate_cue_differences(fmrifeatures_df_2)

    cueB_minus_cueA_1['scannernumber'] = 1
    cueB_minus_cueA_2['scannernumber'] = 2
    
    # Combine the cue differences into the combined dataframe
    combined_cue_differences = pd.concat([cueB_minus_cueA_1, cueB_minus_cueA_2], axis=0, ignore_index=True)
    # Merge the usable outcomes with the cue differences
    combined_df = merge_data(combined_usable_outcomes, combined_cue_differences)
    
    # Merge the demographic data
    combined_df_with_demos = merge_demographics_data(combined_usable_outcomes, combined_cue_differences, file_path)

    #print(combined_df_with_demos.columns)  # This will show you all the column names

    # Prepare the final features (X) and labels (y)
    X_combined, y_combined = prepare_data(combined_df_with_demos)
    
    return X_combined, y_combined


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
    #cueB_minus_cueA = return_cueB_columns(fmrifeatures_df) #CueB only data
    cueB_minus_cueA = return_cueA_columns(fmrifeatures_df) #CueA only data


    
    # Merge the usable outcomes with the cue differences
    merged_df = merge_data(usable_outcomes, cueB_minus_cueA)
    
    # Merge the demographic data
    merged_df_with_demos = merge_demographics_data(usable_outcomes, cueB_minus_cueA, file_path)
    
    # Prepare the final features (X) and labels (y)
    X_demos, y_demos = prepare_data(merged_df_with_demos)

    #print(merged_df_with_demos)
    X_demos, y_demos = prepare_data_with_pca(merged_df_with_demos) #PCA

    
    return X_demos, y_demos


#===============================================================Data preprocessing Finished===============================================================

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


def create_small_model(input_shape, learning_rate=0.001, dropout_rate=0.3):
    model = Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),  # First hidden layer
        layers.Dropout(0.3),  # Dropout for regularization
        layers.Dense(32, activation='relu'),  # Second hidden layer
        layers.Dropout(0.3),  # Dropout for regularization
        layers.Dense(1, activation='sigmoid')  # Output layer (binary classification)
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_deeper_model(input_shape, learning_rate=0.001, dropout_rate=0.3):
    model = Sequential([
        layers.Input(shape=(input_shape,)),
        
        layers.Dense(256),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout_rate),

        layers.Dense(128),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout_rate),

        layers.Dense(64),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout_rate),

        layers.Dense(32),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(dropout_rate),

        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model




def evaluate_model(X, y):
    """Train and evaluate the model with a single train-test split."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)    

    # Apply SMOTE for balancing
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    #MIXUP
    X_train_resampled, y_train_resampled = mixup(X_train_resampled, y_train_resampled, alpha=0.4, n_samples=100)

    # Initialize the model with input shape
    model = create_deeper_model(input_shape=X_train.shape[1])

    # Train the model with resampled data and class weights
    model.fit(
        X_train_resampled,
        y_train_resampled,
        epochs=20,
        batch_size=32,
        verbose=0
    )

    # Evaluate the model on the test set
    results = model.evaluate(X_test, y_test, verbose=0)
    accuracy = results[1]  # Assuming accuracy is the second metric returned

    # Predict probabilities and binary outcomes for the test set
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Calculate AUC and confusion matrix
    auc = roc_auc_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, auc, conf_matrix
    





def run_kfold_cv(file_path, sid, n_splits=10):
    """
    Function to run K-Fold Cross Validation on a dataset.
    
    Parameters:
    - file_path: Path to the dataset.
    - sid: An identifier to decide which dataset to use (1 or 2).
    - n_splits: Number of splits for K-Fold cross-validation (default is 10).
    
    Returns:
    - avg_accuracy: The average accuracy across all folds.
    """
    # Prepare the data
    if(sid==3):
        X_demos, y_demos = process_and_prepare_combined_data(file_path)
    else:
        X_demos, y_demos = process_and_prepare_data(file_path, sid)
    
    # Initialize the K-Fold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # Shuffle and fix random_state for reproducibility
    
    accuracies = []  # To store accuracy scores for each fold
    aucs = []  # To store AUC scores for each fold
    conf_matrices = []  # To store confusion matrices for each fold
    
    # Perform K-Fold Cross Validation
    for train_index, test_index in kf.split(X_demos):
        X_train, X_test = X_demos[train_index], X_demos[test_index]
        y_train, y_test = y_demos[train_index], y_demos[test_index]
        
        # Apply SMOTE for balancing

        
        # Initialize the model with input shape
        model = create_deeper_model(input_shape=X_train.shape[1])
        
        # Train the model with resampled data
        model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate the model on the test set
        results = model.evaluate(X_test, y_test, verbose=0)
        accuracy = results[1]  # Assuming accuracy is the second metric returned
        
        # Predict probabilities and binary outcomes for the test set
        y_pred_prob = model.predict(X_test).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions
        
        # Check if both classes are present in y_test
        if len(Counter(y_test)) > 1:  # If there are at least two different classes
            auc = roc_auc_score(y_test, y_pred_prob)
        else:
            print(f"Warning: Only one class present in y_test for fold. Skipping AUC calculation.")
            auc = None  # AUC is undefined, set to None or handle accordingly
        
        conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
        
        # Store the results
        accuracies.append(accuracy)
        aucs.append(auc)
        conf_matrices.append(conf_matrix)
    
    # Calculate the average accuracy, AUC, and confusion matrix across all folds
    avg_accuracy = np.mean(accuracies)
    avg_auc = np.mean([x for x in aucs if x is not None])  # Handle None values for AUC
    avg_conf_matrix = np.mean(conf_matrices, axis=0)

    print(f"Average Accuracy across {n_splits} folds: {avg_accuracy:.4f}")
    print(f"Average AUC across {n_splits} folds: {avg_auc:.4f}")
    print(f"Average Confusion Matrix across {n_splits} folds:\n{avg_conf_matrix}")
    
    return avg_accuracy, avg_auc, avg_conf_matrix



def base_model_10run_average(file_path):
    """Main function to run the analysis 10 times and return average results."""

    accuracies_with_demo = []
    aucs_with_demo = []
    conf_matrices_with_demo = []

    accuracies_with_demo_ep2 = []
    aucs_with_demo_ep2 = []
    conf_matrices_with_demo_ep2 = []
    runs = 10


    # Run 10 times
    for run in range(runs):
        print(f"Training run {run + 1}/{runs}...")


        #EP1 runs
        X_demos, y_demos = process_and_prepare_data(file_path, 1)
        average_accuracy_demos, auc_demos, conf_matrix_demos = evaluate_model(X_demos, y_demos)
        accuracies_with_demo.append(average_accuracy_demos)
        aucs_with_demo.append(auc_demos)
        conf_matrices_with_demo.append(conf_matrix_demos)


        # EP2 - Same steps for second part of the analysis
        X_demos, y_demos = process_and_prepare_data(file_path, 2)
        average_accuracy_demos, auc_demos, conf_matrix_demos = evaluate_model(X_demos, y_demos)
        accuracies_with_demo_ep2.append(average_accuracy_demos)
        aucs_with_demo_ep2.append(auc_demos)
        conf_matrices_with_demo_ep2.append(conf_matrix_demos)

    # Average results
    avg_accuracy_with_demo = np.mean(accuracies_with_demo)
    avg_auc_with_demo = np.mean(aucs_with_demo)
    avg_conf_matrix_with_demo = np.mean(conf_matrices_with_demo, axis=0)

    avg_accuracy_with_demo_ep2 = np.mean(accuracies_with_demo_ep2)
    avg_auc_with_demo_ep2 = np.mean(aucs_with_demo_ep2)
    avg_conf_matrix_with_demo_ep2 = np.mean(conf_matrices_with_demo_ep2, axis=0)

    print(f"\nEP1 Average Accuracy (with demographics): {avg_accuracy_with_demo:.4f}")
    print(f"EP1 AUC (with demographics): {avg_auc_with_demo:.4f}")
    print("EP1 Confusion Matrix (with demographics):")
    print(avg_conf_matrix_with_demo)

    print(f"\nEP2 Average Accuracy (with demographics): {avg_accuracy_with_demo_ep2:.4f}")
    print(f"EP2 AUC (with demographics): {avg_auc_with_demo_ep2:.4f}")
    print("EP2 Confusion Matrix (with demographics):")
    print(avg_conf_matrix_with_demo_ep2)



def run_multiple_kfolds(file_path, x,sid):
    accuracies = []
    aucs = []
    conf_matrices = []

    for run in range(x):
        print(f"kfold run {run + 1}/{x}...")
        acc, auc, conf_matrix = run_kfold_cv(file_path, sid)
        accuracies.append(acc)
        aucs.append(auc)
        conf_matrices.append(conf_matrix)

    avg_accuracy = np.mean(accuracies)
    avg_auc = np.mean(aucs)
    avg_conf_matrix = np.mean(conf_matrices, axis=0)

    print(f"Average Accuracy over {x} runs: {avg_accuracy:.4f}")
    print(f"Average AUC over {x} runs: {avg_auc:.4f}")
    print("Average Confusion Matrix:")
    print(avg_conf_matrix)

    return avg_accuracy, avg_auc, avg_conf_matrix


#Run the main function with the specified file path
file_path = '/Users/alexd/Documents/Davidson Research/WholeBrainBetas.xlsx'


#base_model_10run_average(file_path)


run_kfold_cv(file_path, 1)
run_kfold_cv(file_path, 2)

#run_multiple_kfolds(file_path,20,1)
#run_multiple_kfolds(file_path,20,2)
#run_multiple_kfolds(file_path,20,3)


#avg_accuracy = run_kfold_cv(file_path, 3)


#base_model_loocv_10run_average(file_path)




#======================================== 10 Fold CV Results EP1 ========================================
# Average Accuracy across 10 folds: 0.6667
# Average AUC across 10 folds: 0.7485
# Average Confusion Matrix across 10 folds:
# [[1.4 1. ]
#  [0.9 2.4]]
#======================================== 10 Fold CV Results EP2 ========================================
# Average Accuracy across 10 folds: 0.6667
# Average AUC across 10 folds: 0.8333
# Average Confusion Matrix across 10 folds:
# [[1.  0.5]
#  [0.7 1.3]]
#======================================== 10 Fold CV Results Combined ========================================
# Average Accuracy across 10 folds: 0.6444
# Average AUC across 10 folds: 0.7990
# Average Confusion Matrix across 10 folds:
# [[2.5 1.4]
#  [1.9 3.4]]


#activation data not correlation data
#best ep1 model is 0.7222
#best ep2 model is 0.6182




#======================================== 10 Fold CV Results just CueA Data EP1 ========================================
# Average Accuracy across 10 folds: 0.7167
# Average AUC across 10 folds: 0.7593
# Average Confusion Matrix across 10 folds:
# [[1.5 0.9]
#  [0.7 2.6]]

#======================================== 10 Fold CV Results just CueA Data EP2 ========================================
# Average Accuracy across 10 folds: 0.6500
# Average AUC across 10 folds: 0.8646
# Average Confusion Matrix across 10 folds:
# [[0.9 0.6]
#  [0.7 1.3]]


#======================================== 10 Fold CV Results just CueB Data EP1   ========================================
# Average Accuracy across 10 folds: 0.7833
# Average AUC across 10 folds: 0.8333
# Average Confusion Matrix across 10 folds:
# [[1.2 0.3]
#  [0.5 1.5]]


#======================================== 10 Fold CV Results just CueB Data EP2  ========================================
# Average Accuracy across 10 folds: 0.6500
# Average AUC across 10 folds: 0.7546
# Average Confusion Matrix across 10 folds:
# [[1.2 1.2]
#  [0.8 2.5]]




#======================================== 10 Fold CV Results EP1 (with pca) deeeper neural network ========================================
# Average Accuracy over 20 runs: 0.6798
# Average AUC over 20 runs: 0.8140
# Average Confusion Matrix:
# [[1.475 0.925]
#  [0.9   2.4  ]]


#======================================== 10 Fold CV Results EP2 (with pca) deeeper neural network ========================================
# Average Accuracy over 20 runs: 0.6521
# Average AUC over 20 runs: 0.7818
# Average Confusion Matrix:
# [[0.91  0.59 ]
#  [0.665 1.335]]

#======================================== 10 Fold CV Results combined (EP1 + EP2) (with pca) deeeper neural network ========================================
# Average Accuracy over 20 runs: 0.6378
# Average AUC over 20 runs: 0.7435
# Average Confusion Matrix:
# [[1.52  2.38 ]
#  [0.985 4.315]]

