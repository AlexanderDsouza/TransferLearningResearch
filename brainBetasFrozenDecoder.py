import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import LeaveOneOut, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE  # Import SMOTE

# TensorFlow/Keras imports
from tensorflow.keras import Input, Model, Sequential, layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.models import load_model





#===================================================DATA PREPROCESSING FUNCTIONS START=====================================================================================

def process_brain_data(file_path, ep_filter):
    """
    Load and process datasets from the specified Excel file.
    - ep_filter: 1 or 2 to filter 'EP1or2' column in 'scannerid' sheet.
    """
    # Load data
    outcome_df = pd.read_excel(file_path, sheet_name='outcomes')
    scanner_ids = pd.read_excel(file_path, sheet_name='scannerid')
    fmrifeatures_df = pd.read_excel(file_path, sheet_name='rawfmrifeatures')
    clinfeatures_df = pd.read_excel(file_path, sheet_name='clinfeatures')
    cueB_minus_cueA = pd.read_excel(file_path, sheet_name='ScannerAdjusted_BminusA')
    demographic_df = pd.read_excel(file_path, sheet_name='demographics')
    
    # Filter scanner IDs based on EP1or2
    subset_scanner_ids = scanner_ids[scanner_ids['EP1or2'] == ep_filter]
    
    # Filter datasets based on SID
    usable_outcomes = outcome_df[outcome_df['SID'].isin(subset_scanner_ids['SID'])]
    fmrifeatures_df = fmrifeatures_df[fmrifeatures_df['SID'].isin(subset_scanner_ids['SID'])]
    clinfeatures_df = clinfeatures_df[clinfeatures_df['SID'].isin(subset_scanner_ids['SID'])]
    demographic_df = demographic_df[demographic_df['SID'].isin(usable_outcomes['SID'])]
    
    # Merge outcomes with clinical features
    usable_outcomes = usable_outcomes.merge(clinfeatures_df, on='SID', how='inner')
    
    # Merge with cue differences
    merged_df = usable_outcomes.merge(cueB_minus_cueA, on='SID', how='inner')
    
    # Merge with demographics
    merged_df = merged_df.merge(demographic_df, on='SID', how='inner')
    
    # Remove duplicate SIDs, keeping the first occurrence
    merged_df = merged_df.drop_duplicates(subset=['SID'], keep='first')

    # Prepare features and labels
    X = merged_df.drop(columns=['Imp20PercentBPRS', 'SID', 'Dx']).values
    y = merged_df['Imp20PercentBPRS'].values
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, merged_df


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
    return X, y



def get_ep1_data_with_demographics(file_path):
    """
    Load and preprocess data for EP1, including demographics.
    
    Args:
        file_path (str): Path to the Excel file containing the data.
    
    Returns:
        tuple: Prepared feature matrix (X) and labels (y) for EP1.
    """
    # Load EP1-specific data
    usable_outcomes, fmrifeatures_df = load_data(file_path)
    
    # Calculate CueB - CueA differences
    cueB_minus_cueA = calculate_cue_differences(fmrifeatures_df)
    
    # Merge outcomes with CueB - CueA differences
    merged_df = merge_data(usable_outcomes, cueB_minus_cueA)
    
    # Merge with demographics data
    merged_with_demos = merge_demographics_data(usable_outcomes, cueB_minus_cueA, file_path)
    
    # Prepare features and labels
    X, y = prepare_data(merged_with_demos)
    
    return X, y

def get_ep2_data_with_demographics(file_path):
    """
    Load and preprocess data for EP1, including demographics.
    
    Args:
        file_path (str): Path to the Excel file containing the data.
    
    Returns:
        tuple: Prepared feature matrix (X) and labels (y) for EP1.
    """
    # Load EP2-specific data
    usable_outcomes, fmrifeatures_df = load_data2(file_path)
    
    # Calculate CueB - CueA differences
    cueB_minus_cueA = calculate_cue_differences(fmrifeatures_df)
    
    # Merge outcomes with CueB - CueA differences
    merged_df = merge_data(usable_outcomes, cueB_minus_cueA)
    
    # Merge with demographics data
    merged_with_demos = merge_demographics_data(usable_outcomes, cueB_minus_cueA, file_path)
    
    # Prepare features and labels
    X, y = prepare_data(merged_with_demos)
    
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

#====================================================DATA PREPROCESSING FUNCTIONS Finished=====================================================================================

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



def process_brain_data(file_path, ep_filter):
    """
    Load and process datasets from the specified Excel file.
    - ep_filter: 1 or 2 to filter 'EP1or2' column in 'scannerid' sheet.
    """
    # Load data
    outcome_df = pd.read_excel(file_path, sheet_name='outcomes')
    scanner_ids = pd.read_excel(file_path, sheet_name='scannerid')
    fmrifeatures_df = pd.read_excel(file_path, sheet_name='rawfmrifeatures')
    clinfeatures_df = pd.read_excel(file_path, sheet_name='clinfeatures')
    cueB_minus_cueA = pd.read_excel(file_path, sheet_name='ScannerAdjusted_BminusA')
    demographic_df = pd.read_excel(file_path, sheet_name='demographics')
    
    # Ensure SID is consistent
    def clean_sid(df):
        df['SID'] = df['SID'].astype(str).str.strip()  # Convert to string and remove whitespace
        return df

    outcome_df = clean_sid(outcome_df)
    scanner_ids = clean_sid(scanner_ids)
    fmrifeatures_df = clean_sid(fmrifeatures_df)
    clinfeatures_df = clean_sid(clinfeatures_df)
    cueB_minus_cueA = clean_sid(cueB_minus_cueA)
    demographic_df = clean_sid(demographic_df)
    
    # Filter scanner IDs based on EP1or2
    subset_scanner_ids = scanner_ids[scanner_ids['EP1or2'] == ep_filter]
    
    # Filter datasets based on SID
    usable_outcomes = outcome_df[outcome_df['SID'].isin(subset_scanner_ids['SID'])]
    fmrifeatures_df = fmrifeatures_df[fmrifeatures_df['SID'].isin(subset_scanner_ids['SID'])]
    clinfeatures_df = clinfeatures_df[clinfeatures_df['SID'].isin(subset_scanner_ids['SID'])]
    demographic_df = demographic_df[demographic_df['SID'].isin(usable_outcomes['SID'])]
    
    # Merge outcomes with clinical features
    merged_df = usable_outcomes.merge(clinfeatures_df, on='SID', how='inner')
    
    # Merge with cue differences
    merged_df = merged_df.merge(cueB_minus_cueA, on='SID', how='inner')

    # Merge with demographics
    merged_df = merged_df.merge(demographic_df, on='SID', how='inner')

    # Remove all SIDs that have duplicates (i.e., keep only unique ones)
    merged_df = merged_df[~merged_df.duplicated(subset=['SID'], keep=False)]  
    
    # Ensure that only SIDs that exist in scanner_ids are kept
    merged_df = merged_df[merged_df['SID'].isin(subset_scanner_ids['SID'])]

    # Prepare features and labels
    X = merged_df.drop(columns=['Imp20PercentBPRS', 'SID', 'Dx'], errors='ignore').values
    y = merged_df['Imp20PercentBPRS'].values
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, merged_df


def evaluate_model(X, y):
    """Perform Leave-One-Out Cross-Validation and evaluate the model with confusion matrix."""
    loo = LeaveOneOut()
    accuracies = []
    predicted_probs = []
    y_true_all = []
    y_pred_all = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Apply SMOTE for balancing
        smote = SMOTE()
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Initialize model with current data shape
        model = create_model(shape=X_train.shape[1])
        
        # Compute class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights[i] for i in np.unique(y_train)}
        
        # Train model with resampled data and class weights
        model.fit(X_train, y_train, epochs=10, batch_size=32, class_weight=class_weight_dict, verbose=0)
        
        # Evaluate model - get only the accuracy
        results = model.evaluate(X_test, y_test, verbose=0)
        accuracy = results[1]  # Extract accuracy assuming it's the second value returned
        accuracies.append(accuracy)
        
        # Predict and store probabilities for AUC calculation
        y_pred_prob = model.predict(X_test).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions
        predicted_probs.append(y_pred_prob[0])
        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])

    # Calculate metrics
    average_accuracy = np.mean(accuracies)
    auc = roc_auc_score(y_true_all, predicted_probs)
    conf_matrix = confusion_matrix(y_true_all, y_pred_all)
    
    return average_accuracy, auc, conf_matrix


#=======================================================two headed model start =========================================================#
def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)
    latent_space = layers.Dense(latent_dim, activation='relu')(x)
    encoder = models.Model(inputs, latent_space, name="encoder")
    return encoder


def build_reconstruction_head(latent_dim, input_shape):
    reconstruction_head = Sequential([
        layers.Input(shape=(latent_dim,)),               # Latent space input
        layers.Dense(128),                               # First dense layer
        layers.LeakyReLU(negative_slope=0.01),           # Non-linear activation
        layers.Dropout(0.3),                             # Dropout for regularization
        layers.Dense(64),                                # Second dense layer
        layers.LeakyReLU(negative_slope=0.01),           # Non-linear activation
        layers.Dropout(0.3),                             # Dropout for regularization
        layers.Dense(input_shape[0], activation='linear')  # Output layer for 174 features
    ], name="reconstruction_head")
    return reconstruction_head

def build_prediction_head(latent_dim):
    prediction_head = Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(128),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(0.3),

        layers.Dense(64),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(0.3),

        layers.Dense(1, activation='sigmoid')
    ],name = 'prediction_head')
    return prediction_head


def train_predictive_reconstruction_model_multiple_runs(
    source_data, latent_dim, runs=10, epochs=20, batch_size=32, test_size=0.3
):
    """
    Train a predictive model with both prediction and reconstruction heads multiple times
    and calculate the average results. After training, return the trained decoder.
    """
    test_losses = []
    test_accuracies = []
    aucs = []
    reconstruction_losses = []
    conf_matrices = []

    decoder = None  # Declare decoder to store trained decoder model

    for run in range(runs):
        print(f"Training run {run + 1}/{runs}...")

        X = source_data.drop(columns=['Imp20PercentBPRS']).values
        y = source_data['Imp20PercentBPRS'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


        #APPLYING MIXUP 
        #print("APPLYING MIXUP")
        #X_train_resampled, y_train_resampled = mixup(X_train_resampled, y_train_resampled, alpha=0.4, n_samples=100)
 
 
        # Get the original feature columns
        #Seeing class distribution
        #class_distribution = Counter(y_train_resampled)
        #print(class_distribution)
        
        input_shape = X_train_resampled.shape[1:]

        encoder = build_encoder(input_shape, latent_dim)
        prediction_head = build_prediction_head(latent_dim)
        reconstruction_head = build_reconstruction_head(latent_dim, input_shape)

        # Combine the encoder with the prediction and reconstruction heads
        latent_representation = encoder(encoder.input)
        prediction_output = prediction_head(latent_representation)
        reconstruction_output = reconstruction_head(latent_representation)

        model = Model(
            inputs=encoder.input,
            outputs={
                'prediction_head': prediction_output,
                'reconstruction_head': reconstruction_output
            },
            name="predictive_reconstruction_model"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'prediction_head': 'binary_crossentropy',
                'reconstruction_head': 'mse'
            },
            metrics={
                'prediction_head': ['accuracy'],
                'reconstruction_head': ['mse']
            }
        )
        #print(y_train_resampled)
        #print(X_train_resampled.shape) 40,173
        # Train the model
        model.fit(
            X_train_resampled,
            {'prediction_head': y_train_resampled, 'reconstruction_head': X_train_resampled},
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # Save the trained decoder (reconstruction head) after each run
        decoder = reconstruction_head  # This stores the trained decoder

        # Evaluate the model
        results = model.evaluate(
            X_test,
            {'prediction_head': y_test, 'reconstruction_head': X_test},
            verbose=0
        )
        test_losses.append(results[0])  # Total loss
        reconstruction_losses.append(results[2])  # Reconstruction loss
        test_accuracies.append(results[3])  # Accuracy for prediction head

        # Predict outcomes and calculate metrics
        y_pred_probs = model.predict(X_test, verbose=0)['prediction_head']
        y_pred = (y_pred_probs > 0.50).astype(int)
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrices.append(conf_matrix)
        auc = roc_auc_score(y_test, y_pred_probs)
        aucs.append(auc)

    # Calculate the average results
    avg_test_loss = np.mean(test_losses)
    avg_test_accuracy = np.mean(test_accuracies)
    avg_auc = np.mean(aucs)
    avg_reconstruction_loss = np.mean(reconstruction_losses)
    avg_conf_matrix = np.mean(conf_matrices, axis=0).astype(int)

    print("=========Results for two headed model test=========")
    print(f"\nAverage Test Loss: {avg_test_loss}")
    print(f"Average Test Accuracy: {avg_test_accuracy}")
    print(f"Average AUC: {avg_auc}")
    print(f"Average Reconstruction Loss: {avg_reconstruction_loss}")
    print("\nAverage Confusion Matrix:")
    print(avg_conf_matrix)


    return decoder  # Return both the results and the trained decoder


#=======================================================two headed model end =======================================================#


def train_predictive_model_multiple_runs(source_data, latent_dim, runs=10, epochs=10, batch_size=32, test_size=0.3):
    """
    Train a predictive model multiple times and calculate the average results.
    
    Args:
        source_data: DataFrame containing the features and target variable.
        latent_dim: Dimension of the latent space for encoding.
        runs: Number of times to train the model.
        epochs: Number of epochs for training the model.
        batch_size: Batch size for training the model.
        test_size: Proportion of the dataset to include in the test split.

    Returns:
        avg_results: Dictionary containing the average evaluation metrics across all runs.
    """
    test_losses = []
    test_accuracies = []
    aucs = []
    conf_matrices = []

    for run in range(runs):
        print(f"Training run {run + 1}/{runs}...")
        
        # Separate features and target
        X = source_data.drop(columns=['Imp20PercentBPRS','SID','Dx'])
        y = source_data['Imp20PercentBPRS']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Balance the training data using SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Build the encoder and prediction head
        encoder = build_encoder(X_train_resampled, latent_dim)
        prediction_head = build_prediction_head(latent_dim)

        #for when not to use encoding before predicting
        prediction_head = build_prediction_head(X_train_resampled.shape[1])  # Use number of features in X_train


        # Create the model by chaining the encoder and prediction head
        model = models.Sequential([
            prediction_head
        ])

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train the model
        model.fit(
            X_train_resampled,
            y_train_resampled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0  # Set to 0 to avoid printing progress
        )

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # Predict outcomes and calculate confusion matrix
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = (y_pred_probs > 0.50).astype(int)  # Adjust threshold as needed
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrices.append(conf_matrix)

        # Additional metrics
        auc = roc_auc_score(y_test, y_pred_probs)
        aucs.append(auc)

    # Calculate the average results
    avg_test_loss = np.mean(test_losses)
    avg_test_accuracy = np.mean(test_accuracies)
    avg_auc = np.mean(aucs)

    # For confusion matrix, compute the average matrix
    avg_conf_matrix = np.mean(conf_matrices, axis=0).astype(int)

    print(f"\nAverage Test Loss: {avg_test_loss}")
    print(f"Average Test Accuracy: {avg_test_accuracy}")
    print(f"Average AUC: {avg_auc}")
    print("\nAverage Confusion Matrix:")
    print(avg_conf_matrix)

    # Return the average results
    avg_results = {
        'avg_test_loss': avg_test_loss,
        'avg_test_accuracy': avg_test_accuracy,
        'avg_auc': avg_auc,
        'avg_confusion_matrix': avg_conf_matrix
    }

    return avg_results



def create_ep2_autoencoder(fmrifeatures_df_EP2, decoder, latent_dim):
    # Create a temporary DataFrame with the correct shape
    temp_df = pd.DataFrame(np.zeros((1, fmrifeatures_df_EP2.shape[1])))
    
    # Freeze the decoder's weights to keep them fixed
    decoder.trainable = False
    
    # Create a new encoder model using the original build_encoder function
     
    input_shape = fmrifeatures_df_EP2.shape[1:]

    encoder = build_encoder(input_shape, latent_dim)

    
    # Create the model
    inputs = layers.Input(shape=(fmrifeatures_df_EP2.shape[1],))
    latent_representation = encoder(inputs)
    reconstructed_output = decoder(latent_representation)
    
    model = models.Model(inputs=inputs, outputs=reconstructed_output)
    return model

def train_ep2_encoder(new_encoder, X, epochs=10, test_size=0.2, batch_size=32):
    # Remove the target column before training
  
    # Split the data into training and testing sets
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)

    # Train the encoder on the training data
    new_encoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Evaluate the encoder on the test data
    test_loss = new_encoder.evaluate(X_test, X_test, verbose=0)
    
    #print(f'Test MSE: {test_loss}')
    
    return new_encoder

    

def encode_ep2data(ep2_encoder, X_ep2):
    """Use the trained encoder to encode the EP2 data."""
    # Drop the target column to get only the features
    # Use the encoder to get encoded features
    encoded_features_ep2 = ep2_encoder.predict(X_ep2)
    return encoded_features_ep2


def train_model_ep2(encoded_features_ep2, y_ep2, input_shape):
    """Train a binary classification model on the encoded features using SMOTE and class weights."""
    
    #debugging purposes 
    #print("Original class distribution:", Counter(y_ep2))

    # Apply SMOTE to balance the classes in the training data
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(encoded_features_ep2, y_ep2)

    #print("Resampled class distribution:", Counter(y_resampled))

    # Define the model with the correct input shape
    model = create_model(input_shape)
    
    # Train the model with class weights
    model.fit(X_resampled, y_resampled, epochs=10, batch_size=32, validation_split=0.2,verbose=0)
    
    return model



def train_and_evaluate_EP2_model(encoded_features_ep2, y_ep2, input_shape, runs=10):
    """Train the model 10 times and return the average accuracy, AUC, and confusion matrix."""
    
    accuracies = []
    aucs = []
    conf_matrices = []
    
    for _ in range(runs):
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            encoded_features_ep2, y_ep2, test_size=0.2, random_state=None, stratify=y_ep2
        )

        # Apply SMOTE to the training set only
        smote = SMOTE(sampling_strategy='auto', random_state=None)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Define the model with the correct input shape
        model = create_model(input_shape)

        # Train the model
        model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

        # Make predictions on the test set
        y_pred, y_pred_prob = predict_ep2_with_model(model, X_test)

        # Evaluate the model on the test set
        accuracy, auc, conf_matrix = evaluate_ep2_model(y_test, y_pred, y_pred_prob)

        # Store results
        accuracies.append(accuracy)
        aucs.append(auc)
        conf_matrices.append(conf_matrix)

    # Compute averages
    avg_accuracy = np.mean(accuracies)
    avg_auc = np.mean(aucs)
    avg_conf_matrix = np.mean(conf_matrices, axis=0)  # Averaging confusion matrices element-wise

    print("=========EP2 Results =========")
    print(f'Accuracy: {accuracy}')
    print(f'AUC: {auc}')
    print('Confusion Matrix:')
    print(conf_matrix)

    return avg_accuracy, avg_auc, avg_conf_matrix

def predict_ep2_with_model(model, encoded_features_ep2, threshold=0.5):
    """Make predictions with the trained model with custom threshold."""
    # Predict probabilities
    y_pred_prob = model.predict(encoded_features_ep2)

    #seeing what values it produces
    #print("Predicted probabilities for the first 20 samples:", y_pred_prob[:20])
    # Convert probabilities to binary predictions using a custom threshold
    y_pred = (y_pred_prob > threshold).astype(int)
    
    return y_pred, y_pred_prob


def evaluate_ep2_model(y_ep2, y_pred, y_pred_prob):
    """Evaluate the model using accuracy, AUC, and confusion matrix."""
    accuracy = accuracy_score(y_ep2, y_pred)
    auc = roc_auc_score(y_ep2, y_pred_prob)
    conf_matrix = confusion_matrix(y_ep2, y_pred)
    

    #print("=========EP2 Results=========")
    #print(f'Accuracy: {accuracy}')
    #print(f'AUC: {auc}')
    #print('Confusion Matrix:')
    #print(conf_matrix)
    
    return accuracy, auc, conf_matrix



def domain_adaptation_simple(file_path):
    
    latent_dim = 64
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
    #some demographics have 2 sids values 
    

    cueB_minus_cueA_EP1 = calculate_cue_differences(fmrifeatures_df_EP1)
    cueB_minus_cueA_EP2 = calculate_cue_differences(fmrifeatures_df_EP2)
    #calculdaing cue differences

    #combining demographics and cueDiffs
    cueB_minus_cueA_EP1 = cueB_minus_cueA_EP1.merge(demographics_df_EP1, on='SID', how='inner')
    cueB_minus_cueA_EP2 = cueB_minus_cueA_EP2.merge(demographics_df_EP2, on='SID', how='inner')



    ep1_data = merge_data(usable_outcomes_EP1, cueB_minus_cueA_EP1)
    ep2_data = merge_data(usable_outcomes_EP2, cueB_minus_cueA_EP2)
    #getting rid of Chg_BPRS label and sid label 


    # Apply MinMax scaling to the data
    scaler = MinMaxScaler()
    ep1_data_normalized = scaler.fit_transform(ep1_data)
    ep2_data_normalized = scaler.transform(ep2_data)  # Use transform on EP2 to avoid data leakage
    ep1_data = pd.DataFrame(ep1_data_normalized, columns=ep1_data.columns)
    ep2_data = pd.DataFrame(ep2_data_normalized, columns=ep2_data.columns)


    # CODE TO PRINT OUT THINGS FOR DEBUGGING IE DATASET/CLASS DIST
    #print(ep1_data)
    #num_rows_bprs_1 = ep1_data[ep1_data['Imp20PercentBPRS'] == 1].shape[0]
    #print(f"Number of rows in ep1 with BPRS = 1: {num_rows_bprs_1}")
    #num_rows_bprs_1 = ep1_data[ep1_data['Imp20PercentBPRS'] == 0].shape[0]
    #print(f"Number of rows in ep1 with BPRS = 0: {num_rows_bprs_1}")

    #num_rows_bprs_2 = ep2_data[ep2_data['Imp20PercentBPRS'] == 1].shape[0]
    #print(f"Number of rows in ep2 with BPRS = 1: {num_rows_bprs_2}")
    #num_rows_bprs_2 = ep2_data[ep2_data['Imp20PercentBPRS'] == 0].shape[0]
    #print(f"Number of rows in ep2 with BPRS = 0: {num_rows_bprs_2}")
    #print(ep1_data.shape)



    #============================Data preprocessing ends=============================================

    # Doing 2 headed model to learn loss reconstruction and prediction loss on EP1
    # this is an initial model run on ep1 to build and save the decoder for ep2


    #print("**************TESTING EP1 DATA WITH PREDICTIVE MODEL**************")
    #train_predictive_model_multiple_runs(ep1_data,latent_dim=64, epochs=20)


    #combined data experiment
    # ep1_data['scannernumber'] = 1
    # ep2_data['scannernumber'] = 2
    # merged_data = pd.concat([ep1_data, ep2_data], ignore_index=True)
    # print("**************TESTING MERGED DATA WITH PREDICTIVE and RECONSTRUCTION MODEL**************")
    # decoder2 = train_predictive_reconstruction_model_multiple_runs(merged_data, latent_dim=64, epochs=20)


    print("**************TESTING EP2 DATA WITH PREDICTIVE and RECONSTRUCTION MODEL**************")
    decoder2 = train_predictive_reconstruction_model_multiple_runs(ep2_data, latent_dim=64, epochs=20)

    print("**************TESTING EP1 DATA WITH PREDICTIVE and RECONSTRUCTION MODEL**************")
    decoder = train_predictive_reconstruction_model_multiple_runs(ep1_data, latent_dim=64, epochs=20)





    #more data preproccessing for training


    y_ep2 = ep2_data['Imp20PercentBPRS']
    X_ep2 = ep2_data.drop(columns=['Imp20PercentBPRS'])

    ep2_encoder = create_ep2_autoencoder(X_ep2, decoder, latent_dim)
    ep2_encoder.compile(optimizer='adam', loss='mse')
    train_ep2_encoder(ep2_encoder,X_ep2)
    encoded_features_ep2 = encode_ep2data(ep2_encoder, X_ep2)

    
    ep2_input_shape = encoded_features_ep2.shape[1]

    
    train_and_evaluate_EP2_model(encoded_features_ep2, y_ep2, ep2_input_shape)
    return






def base_model_10run_average(file_path):
    """Main function to run the analysis 10 times and return average results."""
    accuracies_no_demo = []
    aucs_no_demo = []
    conf_matrices_no_demo = []

    accuracies_with_demo = []
    aucs_with_demo = []
    conf_matrices_with_demo = []

    accuracies_no_demo_ep2 = []
    aucs_no_demo_ep2 = []
    conf_matrices_no_demo_ep2 = []

    accuracies_with_demo_ep2 = []
    aucs_with_demo_ep2 = []
    conf_matrices_with_demo_ep2 = []

    ep1_data,ep2_data = without_subtracting_df(file_path)
    scaler = MinMaxScaler()
    

    y_ep1 = ep1_data['Imp20PercentBPRS']
    X_ep1 = ep1_data.drop(columns=['Imp20PercentBPRS'])
    print(y_ep1)
    print(X_ep1)
    # Apply MinMax scaling to the data
    ep1_data_normalized = scaler.fit_transform(X_ep1)
    ep2_data_normalized = scaler.transform(ep2_data) 


    without_sub_df_ep1_accuracy = []
    without_sub_df_ep1_aucs = []
    without_sub_df_ep1_matrices = []

    # Run 10 times
    for _ in range(10):
        usable_outcomes, fmrifeatures_df = load_data(file_path)
        cueB_minus_cueA = calculate_cue_differences(fmrifeatures_df)
        merged_df = merge_data(usable_outcomes, cueB_minus_cueA)

        X, y = prepare_data(merged_df)
        print(X)
        print(y)


        average_accuracy, auc, conf_matrix = evaluate_model(X_ep1, y_ep1)
        without_sub_df_ep1_accuracy.append(average_accuracy)
        without_sub_df_ep1_aucs.append(auc)
        without_sub_df_ep1_matrices.append(conf_matrix)

        
        
        # Model without demographics
        X, y = prepare_data(merged_df)
        average_accuracy, auc, conf_matrix = evaluate_model(X, y)
        
        # Store the results
        accuracies_no_demo.append(average_accuracy)
        aucs_no_demo.append(auc)
        conf_matrices_no_demo.append(conf_matrix)

        # Model with demographics
        merged_df_with_demos = merge_demographics_data(usable_outcomes, cueB_minus_cueA, file_path)
        X_demos, y_demos = prepare_data(merged_df_with_demos)
        average_accuracy_demos, auc_demos, conf_matrix_demos = evaluate_model(X_demos, y_demos)

        accuracies_with_demo.append(average_accuracy_demos)
        aucs_with_demo.append(auc_demos)
        conf_matrices_with_demo.append(conf_matrix_demos)


        # EP2 - Same steps for second part of the analysis
        usable_outcomes, fmrifeatures_df = load_data2(file_path)
        cueB_minus_cueA = calculate_cue_differences(fmrifeatures_df)
        merged_df = merge_data(usable_outcomes, cueB_minus_cueA)

        # Model without demographics
        X, y = prepare_data(merged_df)
        average_accuracy, auc, conf_matrix = evaluate_model(X, y)

        # Store the results for EP2
        accuracies_no_demo_ep2.append(average_accuracy)
        aucs_no_demo_ep2.append(auc)
        conf_matrices_no_demo_ep2.append(conf_matrix)

        # Model with demographics
        merged_df_with_demos = merge_demographics_data(usable_outcomes, cueB_minus_cueA, file_path)
        X_demos, y_demos = prepare_data(merged_df_with_demos)
        average_accuracy_demos, auc_demos, conf_matrix_demos = evaluate_model(X_demos, y_demos)

        accuracies_with_demo_ep2.append(average_accuracy_demos)
        aucs_with_demo_ep2.append(auc_demos)
        conf_matrices_with_demo_ep2.append(conf_matrix_demos)





    avg_without_sub_df_ep1_accuracy = np.mean(without_sub_df_ep1_accuracy)
    avg_without_sub_df_ep1_aucs = np.mean(without_sub_df_ep1_aucs)
    avg_without_sub_df_ep1_matrices = np.mean(without_sub_df_ep1_matrices, axis=0)


    # Average results
    avg_accuracy_no_demo = np.mean(accuracies_no_demo)
    avg_auc_no_demo = np.mean(aucs_no_demo)
    avg_conf_matrix_no_demo = np.mean(conf_matrices_no_demo, axis=0)

    avg_accuracy_with_demo = np.mean(accuracies_with_demo)
    avg_auc_with_demo = np.mean(aucs_with_demo)
    avg_conf_matrix_with_demo = np.mean(conf_matrices_with_demo, axis=0)

    avg_accuracy_no_demo_ep2 = np.mean(accuracies_no_demo_ep2)
    avg_auc_no_demo_ep2 = np.mean(aucs_no_demo_ep2)
    avg_conf_matrix_no_demo_ep2 = np.mean(conf_matrices_no_demo_ep2, axis=0)

    avg_accuracy_with_demo_ep2 = np.mean(accuracies_with_demo_ep2)
    avg_auc_with_demo_ep2 = np.mean(aucs_with_demo_ep2)
    avg_conf_matrix_with_demo_ep2 = np.mean(conf_matrices_with_demo_ep2, axis=0)


    # Print average results

    print(f"Average  Accuracy (without subtarcting): {avg_without_sub_df_ep1_accuracy:.4f}")
    print(f"AUC (without subtarcting): {avg_without_sub_df_ep1_aucs:.4f}")
    print("Confusion Matrix (without subtarcting):")
    print(avg_without_sub_df_ep1_matrices)
    return


    print(f"Average LOO Accuracy (without demographics): {avg_accuracy_no_demo:.4f}")
    print(f"AUC (without demographics): {avg_auc_no_demo:.4f}")
    print("Confusion Matrix (without demographics):")
    print(avg_conf_matrix_no_demo)

    print(f"\nAverage LOO Accuracy (with demographics): {avg_accuracy_with_demo:.4f}")
    print(f"AUC (with demographics): {avg_auc_with_demo:.4f}")
    print("Confusion Matrix (with demographics):")
    print(avg_conf_matrix_with_demo)

    print(f"\nEP2 Average LOO Accuracy (without demographics): {avg_accuracy_no_demo_ep2:.4f}")
    print(f"EP2 AUC (without demographics): {avg_auc_no_demo_ep2:.4f}")
    print("EP2 Confusion Matrix (without demographics):")
    print(avg_conf_matrix_no_demo_ep2)

    print(f"\nEP2 Average LOO Accuracy (with demographics): {avg_accuracy_with_demo_ep2:.4f}")
    print(f"EP2 AUC (with demographics): {avg_auc_with_demo_ep2:.4f}")
    print("EP2 Confusion Matrix (with demographics):")
    print(avg_conf_matrix_with_demo_ep2)




#Run the main function with the specified file path
file_path = '/Users/alexd/Documents/Davidson Research/WholeBrainBetas.xlsx'

domain_adaptation_simple(file_path)
#base_model_10run_average(file_path)


#============================= BASE MODEL RESULTS =============================
# results over 10 runs
# Average LOO Accuracy (without demographics): 0.6385
# AUC (without demographics): 0.6910
# Confusion Matrix (without demographics):
# [[14.8  8.2]
#  [10.6 18.4]]

# Average LOO Accuracy (with demographics): 0.6596
# AUC (with demographics): 0.7283
# Confusion Matrix (with demographics):
# [[16.   8. ]
#  [11.4 21.6]]

# EP2 Average LOO Accuracy (without demographics): 0.5100
# EP2 AUC (without demographics): 0.5125
# EP2 Confusion Matrix (without demographics):
# [[6.6 5.4]
#  [9.3 8.7]]

# EP2 Average LOO Accuracy (with demographics): 0.6486
# EP2 AUC (with demographics): 0.7007
# EP2 Confusion Matrix (with demographics):
# [[ 9.5  5.5]
#  [ 6.8 13.2]]


#=============================FROZEN DECODER APPROACH=============================
# =========Results for two headed model test=========

# Average Test Loss: 0.7137186467647553
# Average Test Accuracy: 0.575
# Average AUC: 0.6238095238095237
# Average Reconstruction Loss: 0.04007848873734474

# Average Confusion Matrix:
# [[2 4]
#  [2 6]]
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
# =========EP2 Results=========
# Accuracy: 0.6666666666666666
# AUC: 0.5
# Confusion Matrix:
# [[0 2]
#  [0 4]]



#=============================FROZEN DECODER APPROACH (WITH MIXUP)=============================
# Average Test Loss: 0.7656320631504059
# Average Test Accuracy: 0.55625
# Average AUC: 0.6507936507936508
# Average Reconstruction Loss: 0.03214960750192404

# Average Confusion Matrix:
# [[2 4]
#  [2 6]]
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
# =========EP2 Results=========
# Accuracy: 0.6666666666666666
# AUC: 0.5
# Confusion Matrix:
# [[0 2]
#  [0 4]]

#=============================FROZEN DECODER APPROACH (DATASET WIHTOUT SUBTRACTING RAWFMRISHEET)=============================
# =========Results for two headed model test=========

# Average Test Loss: 0.6931818902492524
# Average Test Accuracy: 0.6125
# Average AUC: 0.6634920634920636
# Average Reconstruction Loss: 0.04156188368797302

# Average Confusion Matrix:
# [[3 3]
#  [2 6]]
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
# =========EP2 Results=========
# Accuracy: 0.6666666666666666
# AUC: 0.5
# Confusion Matrix:
# [[0 2]
#  [0 4]]


#=============================FROZEN DECODER APPROACH (DATASET WIHTOUT SUBTRACTING RAWFMRISHEET WITH MIXUP)=============================
# =========Results for two headed model test=========

# Average Test Loss: 0.7699934542179108
# Average Test Accuracy: 0.6125
# Average AUC: 0.6873015873015873
# Average Reconstruction Loss: 0.036048614606261256

# Average Confusion Matrix:
# [[3 3]
#  [2 6]]
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
# =========EP2 Results=========
# Accuracy: 0.6666666666666666
# AUC: 0.5
# Confusion Matrix:
# [[0 2]
#  [0 4]]




#=============================EP2 TOPLINE (results based on twoheaded model)=============================
# =========Results for two headed model test=========

# Average Test Loss: 0.7945817172527313
# Average Test Accuracy: 0.5555555701255799
# Average AUC: 0.5700000000000001
# Average Reconstruction Loss: 0.11058716028928757

# Average Confusion Matrix:
# [[0 3]
#  [0 4]]


#=============================EP2 TOPLINE(WITH MIXUP)=============================
# =========Results for two headed model test=========
# Average Test Loss: 0.8481634318828583
# Average Test Accuracy: 0.5888889044523239
# Average AUC: 0.54
# Average Reconstruction Loss: 0.031036664731800556

# Average Confusion Matrix:
# [[2 1]
#  [2 3]]
