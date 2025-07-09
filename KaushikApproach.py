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



def visualize_tsne(source_data_ep2, encoder, title="t-SNE of Latent Representations"):
    # Extract features and labels
    X = source_data_ep2.drop(columns=['Imp20PercentBPRS']).values
    y = source_data_ep2['Imp20PercentBPRS'].values

    # Encode the data to latent space
    latent_representations = encoder.predict(X)

    # Perform t-SNE on the latent space
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(latent_representations)

    # Create a DataFrame for plotting
    tsne_df = pd.DataFrame({
        'TSNE1': X_tsne[:, 0],
        'TSNE2': X_tsne[:, 1],
        'Label': y
    })

    # Plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Label', palette='coolwarm', s=70, alpha=0.8)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Imp20PercentBPRS')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



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
        conf_matrix = confusion_matrix(y_val, y_pred)
        conf_matrices.append(conf_matrix)
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


def train_predictive_reconstruction_model_multiple_runs(
    ep1_data, latent_dim, runs=10, epochs=20, batch_size=32, test_size=0.3
):
    test_losses = []
    test_accuracies = []
    aucs = []
    conf_matrices = []

    prediction_head = None  # We will return the trained prediction head after training

    for run in range(runs):
        print(f"Training run {run + 1}/{runs}...")

        X = ep1_data.drop(columns=['Imp20PercentBPRS']).values
        y = ep1_data['Imp20PercentBPRS'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
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

        # Train the full model for the entire number of epochs
        model.fit(
            X_train_resampled,
            {'prediction_head': y_train_resampled},
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # Evaluate the trained model
        results = model.evaluate(
            X_test,
            {'prediction_head': y_test},
            verbose=0
        )
        test_losses.append(results[0])
        test_accuracies.append(results[1])

        # Make predictions and calculate metrics
        y_pred_probs = model.predict(X_test, verbose=0)['prediction_head']
        y_pred = (y_pred_probs > 0.50).astype(int)
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrices.append(conf_matrix)
        auc = roc_auc_score(y_test, y_pred_probs)
        aucs.append(auc)

    # Calculate the average results over all runs
    avg_test_loss = np.mean(test_losses)
    avg_test_accuracy = np.mean(test_accuracies)
    avg_auc = np.mean(aucs)
    avg_conf_matrix = np.mean(conf_matrices, axis=0).astype(int)

    print(" =========Results for two headed model test=========")
    print(f"\nAverage Test Loss: {avg_test_loss}")
    print(f"Average Test Accuracy: {avg_test_accuracy}")
    print(f"Average AUC: {avg_auc}")
    print("\nAverage Confusion Matrix:")
    print(avg_conf_matrix)
    return prediction_head  # Return the trained prediction head after all epochs

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
    avg_conf_matrix = np.mean(conf_matrices, axis=0).astype(int)

    print("========= Transfer Learning Results (New Encoder + EP1 Head) =========")
    print(f"\nAverage Loss: {avg_loss}")
    print(f"Average Accuracy: {avg_acc}")
    print(f"Average AUC: {avg_auc}")
    print("\nAverage Confusion Matrix:")
    print(avg_conf_matrix)

    return encoder, ep1_prediction_head, avg_auc, avg_acc



def train_with_new_encoder_and_prediction_head_cv_tsne(
    source_data_ep2, latent_dim, ep1_prediction_head, epochs=20, batch_size=32, n_splits=10,
):
    test_losses = []
    test_accuracies = []
    aucs = []
    conf_matrices = []
    all_embeddings = []
    all_true_labels = []
    all_predictions = []

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = source_data_ep2.drop(columns=['Imp20PercentBPRS']).values
    y = source_data_ep2['Imp20PercentBPRS'].values

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nTraining fold {fold + 1}/{n_splits}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

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

        # Get embeddings for t-SNE
        encoder_output = encoder.predict(X_val)  # Embedding output before the prediction head

        

        # Make predictions
        y_pred_probs = model.predict(X_val, verbose=0)['prediction_head']
        y_pred = (y_pred_probs > 0.5).astype(int)

        # Store embeddings, true labels, and predictions
        all_embeddings.append(encoder_output)
        all_true_labels.append(y_val)
        all_predictions.append(y_pred)

        # Confusion Matrix and AUC
        conf_matrix = confusion_matrix(y_val, y_pred)
        print(conf_matrix)
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

    # Convert lists into arrays for t-SNE
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)


    # Create a boolean mask for correct and incorrect predictions
   # Simplest approach - just color by true label and use different markers for correct/incorrect
    plt.figure(figsize=(12, 10))

    # Create separate scatter plots for each combination of class and correctness
    class0_correct = []
    class0_incorrect = []
    class1_correct = []
    class1_incorrect = []

    for i in range(len(embeddings_2d)):
        is_correct = all_predictions[i] == all_true_labels[i]
        label = all_true_labels[i]
        
        # Sort each point into the appropriate category
        if label == 0 and is_correct:
            class0_correct.append(embeddings_2d[i])
        elif label == 0 and not is_correct:
            class0_incorrect.append(embeddings_2d[i])
        elif label == 1 and is_correct:
            class1_correct.append(embeddings_2d[i])
        elif label == 1 and not is_correct:
            class1_incorrect.append(embeddings_2d[i])

    # Convert lists to numpy arrays for plotting
    class0_correct = np.array(class0_correct) if class0_correct else np.empty((0, 2))
    class0_incorrect = np.array(class0_incorrect) if class0_incorrect else np.empty((0, 2))
    class1_correct = np.array(class1_correct) if class1_correct else np.empty((0, 2))
    class1_incorrect = np.array(class1_incorrect) if class1_incorrect else np.empty((0, 2))

    # Plot each category
    if len(class0_correct) > 0:
        plt.scatter(class0_correct[:, 0], class0_correct[:, 1], 
                    color='blue', marker='o', label="Class 0: Correct", alpha=0.7)
    if len(class0_incorrect) > 0:
        plt.scatter(class0_incorrect[:, 0], class0_incorrect[:, 1], 
                    color='blue', marker='x', label="Class 0: Incorrect", alpha=0.7)
    if len(class1_correct) > 0:
        plt.scatter(class1_correct[:, 0], class1_correct[:, 1], 
                    color='red', marker='o', label="Class 1: Correct", alpha=0.7)
    if len(class1_incorrect) > 0:
        plt.scatter(class1_incorrect[:, 0], class1_incorrect[:, 1], 
                    color='red', marker='x', label="Class 1: Incorrect", alpha=0.7)

    plt.title("t-SNE Visualization: Class Labels and Prediction Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    

    return encoder, ep1_prediction_head, avg_auc, avg_acc


def find_wrong_predictions(
    source_data_ep2, latent_dim, ep1_prediction_head,misclassification_counter,confidence_tracker,risk_tracker, epochs=20, batch_size=32, n_splits=10
):
    test_losses = []
    test_accuracies = []
    aucs = []
    conf_matrices = []
    all_true_labels = []
    all_predictions = []

    confident_accuracies = []
    coverages = []

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = source_data_ep2.drop(columns=['Imp20PercentBPRS']).values
    y = source_data_ep2['Imp20PercentBPRS'].values


    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nTraining fold {fold + 1}/{n_splits}...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)



        input_shape = X_train.shape[1:]

        # Build new encoder
        encoder = build_encoder(input_shape, latent_dim)

        # Optionally freeze prediction head
        ep1_prediction_head.trainable = False  # Or True if you want to fine-tune

        latent_output = encoder(encoder.input)
        prediction_output = ep1_prediction_head(latent_output)

        model = Model(inputs=encoder.input, outputs={'prediction_head': prediction_output}, name="transfer_model")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={'prediction_head': 'binary_crossentropy'},
            metrics={'prediction_head': ['accuracy']}
        )

        # Train only the encoder
        model.fit(
            X_train,
            {'prediction_head': y_train},
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # Evaluate
        results = model.evaluate(X_val, {'prediction_head': y_val}, verbose=0)
        test_losses.append(results[0])
        test_accuracies.append(results[1])

        # Make predictions
        y_pred_probs = model.predict(X_val, verbose=0)['prediction_head']
        y_pred = (y_pred_probs > 0.5).astype(int)

        for idx, prob, true_label in zip(val_idx, y_pred_probs.flatten(), y_val):
            confidence = 2 * max(prob, 1 - prob) - 1  # Now in [0, 1]
            risk = 1 - confidence  # Also in [0, 1]
            print(prob)
            confidence_tracker[idx].append(confidence)
            risk_tracker[idx].append(risk)
        
        
        y_val = y_val.flatten()  # Ensure y_val is a 1D array
        y_pred = y_pred.flatten()  # Ensure y_pred is a 1D array   

        confidence_threshold = 0.6
        confident_indices = np.where((y_pred_probs >= confidence_threshold) | (y_pred_probs <= (1 - confidence_threshold)))[0]

        if len(confident_indices) > 0:
            confident_preds = y_pred[confident_indices]
            confident_labels = y_val[confident_indices]
            confident_accuracy = accuracy_score(confident_labels, confident_preds)
            confident_accuracies.append(confident_accuracy)

            coverage = len(confident_indices) / len(y_val)
            coverages.append(coverage)
        else:
            confident_accuracies.append(np.nan)  # or 0.0
            coverages.append(0.0)

        #print(y_pred)
        #print(y_val)

        incorrect_val_indices = val_idx[np.where(y_pred != y_val)[0]]
        #print("Incorrect indices this fold:", incorrect_val_indices.tolist())

        # Update the counter with the indices of misclassified points
        misclassification_counter.update(incorrect_val_indices)

        # Confusion Matrix and AUC
        conf_matrix = confusion_matrix(y_val, y_pred)
        #print(conf_matrix)
        conf_matrices.append(conf_matrix)
        auc = roc_auc_score(y_val, y_pred_probs)
        aucs.append(auc)
        # Average metrics

    avg_confident_accuracy = np.nanmean(confident_accuracies)  # Use nanmean in case of empty folds
    avg_coverage = np.mean(coverages)
    
    avg_loss = np.mean(test_losses)
    avg_acc = np.mean(test_accuracies)
    avg_auc = np.mean(aucs)
    avg_conf_matrix = np.mean(conf_matrices, axis=0)






    return encoder, ep1_prediction_head, avg_auc, avg_acc,avg_confident_accuracy,avg_coverage



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

    #=============================================Data preprocessing ends=============================================
    
    avg_tracker = []
    auc_tracker = []

    prediction_head = train_predictive_reconstruction_model_with_kfold(ep1_data, latent_dim=64, epochs=20, batch_size=32, num_folds=5)
    #train_with_new_encoder_and_prediction_head_cv(source_data_ep2=ep2_data,latent_dim=64,ep1_prediction_head=prediction_head,epochs=20,batch_size=32,n_splits=10)
    #train_with_new_encoder_and_prediction_head_cv_tsne(source_data_ep2=ep2_data,latent_dim=64,ep1_prediction_head=prediction_head,epochs=20,batch_size=32,n_splits=10)

    total_aucs = []
    total_accuracies = []


    
    confidence_tracker = defaultdict(list)
    risk_tracker = defaultdict(list)
    misclassification_counter = Counter()
    runs = 100
    for i in range(runs):
        print(f"Run {i+1}/{runs}")
        encoder_ep2, prediction_head, auc, acc,conf_acc,coverage = find_wrong_predictions(
            source_data_ep2=ep2_data,
            latent_dim=64,
            ep1_prediction_head=prediction_head,
            epochs=20,
            batch_size=32,
            n_splits=10,
            misclassification_counter=misclassification_counter,
            confidence_tracker = confidence_tracker,
            risk_tracker = risk_tracker
        )
        total_aucs.append(auc)
        total_accuracies.append(acc)



    all_indices = list(range(len(ep2_data)))


    # misclassification plot 
    misclass_counts = [misclassification_counter.get(i, 0) for i in all_indices]
    plt.figure(figsize=(14, 6))
    plt.bar(all_indices, misclass_counts, color='salmon', edgecolor='black')
    plt.title("Misclassifications per Data Point")
    plt.xlabel("Data Point Index")
    plt.ylabel("Number of Times Misclassified")
    plt.grid(True, linestyle='--', alpha=0.5)
    #desktop_path = os.path.expanduser("~/Desktop/misclassifications_plot_1000.png")
    # Save the plot
    #plt.savefig(desktop_path)
    plt.tight_layout()
    plt.show()


    avg_conf_error = {idx: np.mean(confidence_tracker[idx]) for idx in confidence_tracker}

    # Sorted indices for consistent plotting
    sorted_indices = sorted(avg_conf_error.keys())

    # Prepare data
    conf_error_values = [avg_conf_error[idx] for idx in sorted_indices]
    misclass_freq = [misclassification_counter.get(idx, 0) for idx in sorted_indices]

    # Normalize misclassification count for coloring
    norm = plt.Normalize(min(misclass_freq), max(misclass_freq))
    colors = plt.cm.Reds(norm(misclass_freq))

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(12, 5))
    scatter = ax.scatter(sorted_indices, conf_error_values, c=misclass_freq, cmap='Reds', edgecolor='black')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array([])  # Avoid warning
    fig.colorbar(sm, ax=ax, label='Times Misclassified')

    # Labels and styling
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Average Confidence Error (abs(ŷ - y))')
    ax.set_title('Prediction Confidence Error vs Misclassification Frequency')
    ax.set_ylim(0, 1)
    plt.tight_layout()

    # Save and show
    desktop_path = os.path.expanduser("~/Desktop/confidence_and_misclassification.png")
    plt.savefig(desktop_path)
    plt.show()


    # Risk plot
    avg_risk = {idx: np.mean(risk_tracker[idx]) for idx in risk_tracker}
    sorted_indices = sorted(avg_risk.keys())
    risk_values = [avg_risk[idx] for idx in sorted_indices]
    misclass_freq = [misclassification_counter.get(idx, 0) for idx in sorted_indices]
    norm = plt.Normalize(min(misclass_freq), max(misclass_freq))
    colors = plt.cm.Blues(norm(misclass_freq))  # Changed color map to Blues for risk
    fig, ax = plt.subplots(figsize=(12, 5))  # Create a figure and axis
    bars = ax.bar(sorted_indices, risk_values, color=colors, edgecolor='black')
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
    sm.set_array([])  # To avoid warning about empty data
    fig.colorbar(sm, ax=ax, label='Times Misclassified')
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Average Risk (min(ŷ, 1-ŷ))')
    ax.set_title('Prediction Risk vs Misclassification Frequency')
    desktop_path = os.path.expanduser("~/Desktop/risk_and_misclassification.png")
    ax.set_ylim(0, 1)  # Risk ranges from 0 to 0.5
    plt.savefig(desktop_path)
    plt.tight_layout()
    plt.show()


    overall_avg_auc = np.mean(total_aucs)
    overall_avg_accuracy = np.mean(total_accuracies)

    print("\n========== FINAL AVERAGES ==========")
    print(f"Average AUC over {runs} runs: {overall_avg_auc}")
    print(f"Average Accuracy over {runs} runs: {overall_avg_accuracy}")
    print(f"Average Accuracy with confidence filter over {runs} runs: {conf_acc}")
    print(f"Average coverage over {runs} runs: {coverage}")



    return

        #just getting averages
    for i in range(10):
        print(f"\n======= Run {i+1}/10 =======")
        encoder_ep2, prediction_head, auc, acc = train_with_new_encoder_and_prediction_head_cv(
            source_data_ep2=ep2_data,
            latent_dim=64,
            ep1_prediction_head=prediction_head,
            epochs=20,
            batch_size=32,
            n_splits=10
        )
        total_aucs.append(auc)
        total_accuracies.append(acc)

    overall_avg_auc = np.mean(total_aucs)
    overall_avg_accuracy = np.mean(total_accuracies)

    print("\n========== FINAL AVERAGES ==========")
    print(f"Average AUC over 10 runs: {overall_avg_auc}")
    print(f"Average Accuracy over 10 runs: {overall_avg_accuracy}")

    return 





file_path = '/Users/alexd/Documents/Davidson Research/WholeBrainBetas.xlsx'
domain_adaptation_simple(file_path)


#RESULTS KAUSHIK APPROACH GOOD RUN.
#  =========Results for two headed model test=========

# Average Test Loss: 0.6792838752269745
# Average Test Accuracy: 0.55
# Average AUC: 0.5873015873015873

# Average Confusion Matrix:
# [[3 3]
#  [3 5]]
# =========EP2 Results=========

# Average Test Loss: 0.6819887161254883
# Average Test Accuracy: 0.6363636255264282
# Average AUC: 0.5999999999999999

# Average Confusion Matrix:
# [[2 3]
#  [1 5]]


#Two headed model test on EP2 as topline
# Average Test Loss: 0.7249949216842652
# Average Test Accuracy: 0.5636363685131073
# Average AUC: 0.5199999999999998

# Average Confusion Matrix:
# [[2 2]
#  [1 4]]


# =========Combined Dataset Results=========
# Average Test Loss: 0.707160472869873
# Average Test Accuracy: 0.5192307740449905
# Average AUC: 0.5006060606060606

# Average Confusion Matrix:
# [[4 6]
#  [6 8]]


#Averages
# Average Accuracy over 10 different : 0.509090918302536
# Variance Accuracy: 0.0052892549964023055
# Average AUC: 0.509090918302536
# Variance AUC: 0.0052892549964023055





# ========== FINAL AVERAGES CUEA data ==========
# Average AUC over 10 runs: 0.5625000000000001
# Average Accuracy over 10 runs: 0.5133333399891853


# ========== FINAL AVERAGES CUEB data==========
# Average AUC over 10 runs: 0.645
# Average Accuracy over 10 runs: 0.5816666749119758

# ========== FINAL AVERAGES cueb minus cuea ==========
# Average AUC over 10 runs: 0.5349999999999999
# Average Accuracy over 10 runs: 0.5308333408832551
