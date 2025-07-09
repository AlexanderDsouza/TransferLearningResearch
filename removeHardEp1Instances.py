import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


from collections import Counter
from collections import defaultdict


from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import Input, Model 
from tensorflow.keras import models
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.models import load_model

from keras.layers import Input, Dense

from sklearn.manifold import TSNE
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from scipy.stats import zscore

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np



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

def merge_data_keep_sid(usable_outcomes, cueB_minus_cueA):
    """Merge usable outcomes with CueB - CueA differences and remove identical rows."""
    merged_df_with_SID = usable_outcomes.merge(cueB_minus_cueA, on='SID', how='inner')
    merged_df_with_SID = merged_df_with_SID.drop_duplicates()
    return merged_df_with_SID.drop(columns=['Chg_BPRS'])

#====================================================DATA PREPROCESSING Finished=====================================================================================

#=======================================================two headed model start =========================================================#

def build_encoder(input_shape, latent_dim):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = layers.LayerNormalization()(x)
    
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.LayerNormalization()(x)
    
    latent_space = layers.Dense(latent_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    
    encoder = tf.keras.models.Model(inputs, latent_space, name="encoder")
    return encoder


def build_prediction_head(latent_dim):
    prediction_head = tf.keras.Sequential([
        layers.Input(shape=(latent_dim,)),

        layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.LayerNormalization(),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(0.3),

        layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.LayerNormalization(),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(0.3),

        layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ], name='prediction_head')

    return prediction_head




def train_predictive_reconstruction_model_with_kfold(ep1_data, latent_dim, epochs=20, batch_size=32, num_folds=5):
    test_losses = []
    test_accuracies = []
    aucs = []
    conf_matrices = []

    # KFold cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    for fold, (train_index, val_index) in enumerate(kf.split(ep1_data)):
        print(f"Training fold {fold + 1}/{num_folds}...")

        # Split the data into training and validation sets
        train_data, val_data = ep1_data.iloc[train_index], ep1_data.iloc[val_index]
        X_train = train_data.drop(columns=['Imp20PercentBPRS']).values
        y_train = train_data['Imp20PercentBPRS'].values
        X_val = val_data.drop(columns=['Imp20PercentBPRS']).values
        y_val = val_data['Imp20PercentBPRS'].values

        # Resample the training data using SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        input_shape = X_train_resampled.shape[1:]

        # Build the encoder and prediction head
        encoder = build_encoder(input_shape, latent_dim)
        prediction_head = build_prediction_head(latent_dim)

        # Combine the encoder with the prediction head
        latent_representation = encoder(encoder.input)
        prediction_output = prediction_head(latent_representation)

        # Train the model with encoder + prediction head
        model = Model(
            inputs=encoder.input,
            outputs={'prediction_head': prediction_output},
            name="predictive_model"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={'prediction_head': 'binary_crossentropy'},
            metrics={'prediction_head': ['accuracy']}
        )

        # Train the model
        model.fit(
            X_train_resampled,
            {'prediction_head': y_train_resampled},
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # Evaluate the model on the validation set
        results = model.evaluate(X_val, {'prediction_head': y_val}, verbose=0)
        test_losses.append(results[0])
        test_accuracies.append(results[1])

        # Make predictions and calculate metrics
        y_pred_probs = model.predict(X_val, verbose=0)['prediction_head']
        y_pred = (y_pred_probs > 0.50).astype(int)
        conf_matrix = confusion_matrix(y_val, y_pred, labels=[0,1])
        conf_matrices.append(conf_matrix)
        if len(np.unique(y_val)) < 2:
            print(f"[WARN] Only one class present in y_val: {np.unique(y_val)}. Skipping AUC for this fold.")
            continue  # or set auc = None or 0.5 depending on your logic
        else:
            auc = roc_auc_score(y_val, y_pred_probs)
            aucs.append(auc)

    # Calculate the average results over all folds
    avg_test_loss = np.mean(test_losses)
    avg_test_accuracy = np.mean(test_accuracies)
    avg_auc = np.mean(aucs)
    avg_conf_matrix = np.mean(conf_matrices, axis=0)

    print(" =========Results for 5-Fold Cross Validation=========")
    print(f"\nAverage Test Loss: {avg_test_loss}")
    print(f"Average Test Accuracy: {avg_test_accuracy}")
    print(f"Average AUC: {avg_auc}")
    print("\nAverage Confusion Matrix:")
    print(avg_conf_matrix)

    return prediction_head  # Return the trained prediction head after cross-validation


#=======================================================two headed model end =======================================================#
def train_with_new_encoder_and_prediction_head_cv(
    source_data_ep2, latent_dim, ep1_prediction_head, epochs=20, batch_size=32, n_splits=10
):
    test_losses = []
    test_accuracies = []
    aucs = []
    conf_matrices = []

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = source_data_ep2.drop(columns=['Imp20PercentBPRS']).values
    y = source_data_ep2['Imp20PercentBPRS'].values

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nTraining fold {fold + 1}/{n_splits}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # SMOTE Resampling
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        input_shape = X_train_resampled.shape[1:]

        # Build new encoder
        encoder = build_encoder(input_shape, latent_dim)

        # Optionally freeze prediction head (if you want to fine-tune only encoder)
        ep1_prediction_head.trainable = False  # Or True if you want to fine-tune

        latent_output = encoder(encoder.input)
        prediction_output = ep1_prediction_head(latent_output)

        model = Model(inputs=encoder.input, outputs={'prediction_head': prediction_output}, name="transfer_model")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={'prediction_head': 'binary_crossentropy'},
            metrics={'prediction_head': ['accuracy']}
        )

        # Train only the encoder (head is frozen or reused)
        model.fit(
            X_train_resampled,
            {'prediction_head': y_train_resampled},
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # Evaluate
        results = model.evaluate(X_val, {'prediction_head': y_val}, verbose=0)
        test_losses.append(results[0])
        test_accuracies.append(results[1])

        y_pred_probs = model.predict(X_val, verbose=0)['prediction_head']
        y_pred = (y_pred_probs > 0.5).astype(int)
        conf_matrix = confusion_matrix(y_val, y_pred)
        conf_matrices.append(conf_matrix)
        auc = roc_auc_score(y_val, y_pred_probs)
        aucs.append(auc)




    # Average metrics
    avg_loss = np.mean(test_losses)
    avg_acc = np.mean(test_accuracies)
    avg_auc = np.mean(aucs)
    avg_conf_matrix = np.mean(conf_matrices, axis=0)

    print("========= Transfer Learning Results (New Encoder + EP1 Head) =========")
    print(f"\nAverage Loss: {avg_loss}")
    print(f"Average Accuracy: {avg_acc}")
    print(f"Average AUC: {avg_auc}")
    print("\nAverage Confusion Matrix:")
    print(avg_conf_matrix)

    return encoder, ep1_prediction_head, avg_auc, avg_acc


def get_misclassification_rates_per_sid(data, model_fn, runs=100, latent_dim=64, epochs=20, batch_size=32, num_folds=5):
    """
    Train the model 'runs' times on ep1_data_sid, track misclassification per sid.
    model_fn: a function that trains model and returns predictions & true labels with sids for validation folds.
    """


    # Dict to accumulate misclassification counts and total counts per sid
    sid_misclass_counts = defaultdict(int)
    sid_total_counts = defaultdict(int)

    for run in range(runs):
        print(f"Run {run+1}/{runs}")

        # This function trains on data with k-fold, returns list of (y_true, y_pred, sid) per fold
        fold_results = model_fn(data, latent_dim=latent_dim, epochs=epochs, batch_size=batch_size, num_folds=num_folds)

        # fold_results: list of tuples per fold [(y_true_fold, y_pred_fold, sid_fold), ...]
        for y_true, y_pred, sid_arr in fold_results:
            for true_label, pred_label, sid in zip(y_true, y_pred, sid_arr):
                sid_total_counts[sid] += 1
                if pred_label != true_label:
                    sid_misclass_counts[sid] += 1

    # Compute misclassification rate per sid
    sid_misclassification_rate = {sid: sid_misclass_counts[sid] / sid_total_counts[sid] for sid in sid_total_counts}

    # Convert to DataFrame for easier handling
    df_misclass = pd.DataFrame({
        'SID': list(sid_misclassification_rate.keys()),
        'misclassification_rate': list(sid_misclassification_rate.values())
    })

    # ✅ Save to CSV
    df_misclass.to_csv('misclassification_rates_per_sid.csv', index=False)
    print("✅ Misclassification rates saved to 'misclassification_rates_per_sid.csv'")

    return df_misclass


def model_fn(data, latent_dim=64, epochs=20, batch_size=32, num_folds=5):
    from sklearn.model_selection import StratifiedKFold

    results = []

    X = data.drop(columns=['Imp20PercentBPRS', 'SID']).values
    y = data['Imp20PercentBPRS'].values
    sids = data['SID'].values

    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=None)

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        sid_val = sids[val_idx]

        # SMOTE resampling on training set
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        if len(X_train_resampled) == 0 or len(y_train_resampled) == 0:
            print("Empty training set after SMOTE. Skipping this fold.")
            continue
        input_shape = X_train_resampled.shape[1:]

        # Build encoder and prediction head
        encoder = build_encoder(input_shape, latent_dim)
        prediction_head = build_prediction_head(latent_dim)

        # Connect encoder + prediction head
        latent_output = encoder(encoder.input)
        prediction_output = prediction_head(latent_output)

        model = Model(inputs=encoder.input, outputs=prediction_output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        model.fit(
            X_train_resampled,
            y_train_resampled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # Predict on validation set
        y_pred_probs = model.predict(X_val, verbose=0).flatten()
        y_pred = (y_pred_probs > 0.5).astype(int)

        # Append y_true, y_pred, sid_val for misclassification calculation later
        results.append((y_val, y_pred, sid_val))

    return results



def domain_adaptation_simple(file_path):
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

    #cueB_minus_cueA_EP1 = return_cueA_columns(fmrifeatures_df_EP1) #CueA only data
    #cueB_minus_cueA_EP2 = return_cueA_columns(fmrifeatures_df_EP2) #CueA only data

    cueB_minus_cueA_EP1 = return_cueB_columns(fmrifeatures_df_EP1) #CueB only data
    cueB_minus_cueA_EP2 = return_cueB_columns(fmrifeatures_df_EP2) #CueB only data

    #calculdaing cue differences

    #combining demographics and cueDiffs
    cueB_minus_cueA_EP1 = cueB_minus_cueA_EP1.merge(demographics_df_EP1, on='SID', how='inner')
    cueB_minus_cueA_EP2 = cueB_minus_cueA_EP2.merge(demographics_df_EP2, on='SID', how='inner')

    ep1_data = merge_data(usable_outcomes_EP1, cueB_minus_cueA_EP1)
    ep2_data = merge_data(usable_outcomes_EP2, cueB_minus_cueA_EP2)

    ep1_data_sid = merge_data_keep_sid(usable_outcomes_EP1, cueB_minus_cueA_EP1)
    ep2_data_sid = merge_data_keep_sid(usable_outcomes_EP2, cueB_minus_cueA_EP2)

    #=============================================Data preprocessing ends=============================================
    
    avg_tracker = []
    auc_tracker = []


# Load misclassification rates from CSV

    df_misclass = pd.read_csv("sid_misclass_summary2.csv")  # Adjust path if needed



    # Sort by misclassification rate for better visualization
    df_sorted = df_misclass.sort_values(by='misclass_percent', ascending=False)

    # Create the bar plot
    plt.figure(figsize=(14, 6))
    plt.bar(df_sorted['SID'].astype(str), df_sorted['misclass_percent'], color='skyblue')

    # Customize the plot
    plt.xlabel("SID")
    plt.ylabel("Misclassification Rate")
    plt.title("Misclassification Rate per SID (EP2)")
    plt.xticks(rotation=90)
    plt.tight_layout()

    desktop_path = os.path.expanduser("~/Desktop/missclassification_rates_ep2.png")
    plt.savefig(desktop_path)
    # Show the plot
    plt.show()
    



    df_misclass = pd.read_csv("ep1_misclass_rates.csv")  # Adjust path if needed



    # Sort by misclassification rate for better visualization
    df_sorted = df_misclass.sort_values(by='misclassification_rate', ascending=False)

    # Create the bar plot
    plt.figure(figsize=(14, 6))
    plt.bar(df_sorted['SID'].astype(str), df_sorted['misclassification_rate'], color='skyblue')

    # Customize the plot
    plt.xlabel("SID")
    plt.ylabel("Misclassification Rate")
    plt.title("Misclassification Rate per SID (EP1)")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Show the plot
    desktop_path = os.path.expanduser("~/Desktop/missclassification_rates_ep1.png")
    plt.savefig(desktop_path)
    plt.show()


    
    # Compute threshold and get hard SIDs
    threshold = df_misclass['misclassification_rate'].quantile(0.7)
    hard_sids = df_misclass[df_misclass['misclassification_rate'] >= threshold]['SID'].values

    # Drop the hardest instances from ep1_data_sid
    easy_instances = ep1_data_sid[~ep1_data_sid['SID'].isin(hard_sids)]

    # Drop sid before training


    ep1_data = easy_instances.drop(columns=['SID'])


    prediction_head = train_predictive_reconstruction_model_with_kfold(ep1_data, latent_dim=64, epochs=20, batch_size=32, num_folds=5)
    train_with_new_encoder_and_prediction_head_cv(source_data_ep2=ep2_data,latent_dim=64,ep1_prediction_head=prediction_head,epochs=20,batch_size=32,n_splits=10)

    total_aucs = []
    total_accuracies = []


    return






file_path = '/Users/alexd/Documents/Davidson Research/WholeBrainBetas.xlsx'
domain_adaptation_simple(file_path)

