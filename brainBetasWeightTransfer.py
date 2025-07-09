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



from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.metrics import Accuracy



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


def prepare_data(merged_df):
    """Prepare features and labels from the merged DataFrame."""
    X = merged_df.drop(columns=['Imp20PercentBPRS']).values
    y = merged_df['Imp20PercentBPRS'].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

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




def evaluate_model_no_loocv(X, y):
    """Train and evaluate the model with a single train-test split."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)    

    # Apply SMOTE for balancing
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    #MIXUP
    X_train_resampled, y_train_resampled = mixup(X_train_resampled, y_train_resampled, alpha=0.4, n_samples=100)

    # Initialize the model with input shape
    model = create_model(input_shape=X_train.shape[1])

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


def evaluate_model_no_loocv_ep2(X, y, weights_file_path):
    """Train and evaluate the model for ep2 with smaller learning rate and frozen layers."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)    

    # Apply SMOTE for balancing
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # MIXUP
    X_train_resampled, y_train_resampled = mixup(X_train_resampled, y_train_resampled, alpha=0.4, n_samples=100)

    # Initialize the model for ep2
    model = create_model(input_shape=X_train.shape[1])

    # Load the weights from ep1
    model.load_weights(weights_file_path)

    # Freeze the layers you want to keep unchanged (optional)
    # for layer in model.layers:
    #     layer.trainable = False  # Freeze all layers initially
    
    # # Unfreeze the last layers to train them
    # for layer in model.layers[-2:]:  # Example: Only unfreeze the last 2 layers
    #     layer.trainable = True

    # Compile the model with a smaller learning rate
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

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



def save_model_weights(model, file_path):
    """Save the model's weights to a file."""
    model.save_weights(file_path)

def load_model_weights(model, file_path):
    """Load the model's weights from a file."""
    model.load_weights(file_path)

def base_model_no_loocv_10run_average(file_path):
    """Train ep1 model once, save weights, and use ep1's weights for predictions on ep2 10 times."""
    accuracies_with_demo = []
    aucs_with_demo = []
    conf_matrices_with_demo = []

    # Set the path to save the weights for ep1
    weights_file_path = "/Users/alexd/Documents/Davidson Research/ep1_weights.weights.h5"

    # 1. Train ep1 model once and save weights
    print("Training ep1 model...")

    # Load and prepare ep1 data
    usable_outcomes, fmrifeatures_df = load_data(file_path)
    cueB_minus_cueA = calculate_cue_differences(fmrifeatures_df)
    merged_df = merge_data(usable_outcomes, cueB_minus_cueA)

    # Prepare data for training ep1
    merged_df_with_demos = merge_demographics_data(usable_outcomes, cueB_minus_cueA, file_path)
    X_demos, y_demos = prepare_data(merged_df_with_demos)

    # Create and train the ep1 model
    model_ep1 = create_model(input_shape=X_demos.shape[1])
    model_ep1.fit(X_demos, y_demos, epochs=20, batch_size=32, verbose=0)
    
    accuracy, auc, conf_matrix = evaluate_model_no_loocv(X_demos, y_demos)
    print(f"accuracy for ep1 {accuracy}")

    # Save the weights for ep1
    save_model_weights(model_ep1, weights_file_path)
    print("ep1 model trained and weights saved.")

    # 2. Use the saved ep1 model for 10 predictions on ep2 data
    runs = 10
    for run in range(runs):
        print(f"Prediction run {run + 1}/{runs}...")

        # Load the weights for ep1 model (we load once for each prediction)
        model_ep2 = create_model(input_shape=X_demos.shape[1])  # Create a new model for ep2
        model_ep2.load_weights(weights_file_path)  # Load the saved ep1 weights into model_ep2

        # Load and prepare ep2 data
        usable_outcomes_ep2, fmrifeatures_df_ep2 = load_data2(file_path)  # Assuming it's the same file, change if needed
        cueB_minus_cueA_ep2 = calculate_cue_differences(fmrifeatures_df_ep2)
        merged_df_ep2 = merge_data(usable_outcomes_ep2, cueB_minus_cueA_ep2)
        # Prepare ep2 data for prediction
        merged_df_with_demos_ep2 = merge_demographics_data(usable_outcomes_ep2, cueB_minus_cueA_ep2, file_path)
        X_demos_ep2, y_demos_ep2 = prepare_data(merged_df_with_demos_ep2)

        # Predict using the model (ep2 model using ep1's weights)
        predictions = model_ep2.predict(X_demos_ep2)

        # Use evaluate_model_no_loocv to calculate metrics for each run
        accuracy, auc, conf_matrix = evaluate_model_no_loocv_ep2(X_demos_ep2, y_demos_ep2, weights_file_path)

        # Store metrics for averaging
        print(f"Accuracy for run {run} = {accuracy}")
        accuracies_with_demo.append(accuracy)
        aucs_with_demo.append(auc)
        conf_matrices_with_demo.append(conf_matrix)

    # Average results
    avg_accuracy_with_demo = np.mean(accuracies_with_demo)
    avg_auc_with_demo = np.mean(aucs_with_demo)
    avg_conf_matrix_with_demo = np.mean(conf_matrices_with_demo, axis=0)

    print(f"\nAverage Accuracy (with demographics): {avg_accuracy_with_demo:.4f}")
    print(f"AUC (with demographics): {avg_auc_with_demo:.4f}")
    print("Confusion Matrix (with demographics):")
    print(avg_conf_matrix_with_demo)


#Run the main function with the specified file path
file_path = '/Users/alexd/Documents/Davidson Research/WholeBrainBetas.xlsx'

#domain_adaptation_simple(file_path)

base_model_no_loocv_10run_average(file_path)
#base_model_loocv_10run_average(file_path)
#main_with_pca(file_path, n_components=33)





# weight transfer results
#accuracy for ep1 0.7222222089767456
# Average Accuracy (with demographics): 0.5455
# AUC (with demographics): 0.5633
# Confusion Matrix (with demographics):
# [[2. 3.]
#  [2. 4.]]

