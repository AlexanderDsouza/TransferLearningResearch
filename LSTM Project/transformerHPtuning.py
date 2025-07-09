
# Core libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy
import csv
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import os

# SciPy for statistical analysis
from scipy import stats
from scipy.stats import t
from tensorflow.keras.optimizers import Adam

# Scikit-learn utilities
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, balanced_accuracy_score, ConfusionMatrixDisplay, classification_report
)

import matplotlib.pyplot as plt

from sklearn.utils import resample, class_weight
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer

# TensorFlow and Keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Concatenate,
    LayerNormalization, MultiHeadAttention, Add
)
import keras.backend as K

from tensorflow.keras.callbacks import EarlyStopping
# SHAP for explainability
import shap


def focal_loss(gamma=2., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)        
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        loss = weight * cross_entropy
        return tf.reduce_mean(loss)
    
    return focal_loss_fixed

def transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([inputs, x])

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([res, x])


# ===========================
# Load and preprocess dataset
# ===========================
df = pd.read_csv('for_JS_final_withgroup.csv')

df['site'] = df['site_id_l.baseline_year_1_arm_1'].str.extract(r'(\d+)$').astype(int)

features_baseline = [
    #'interview_age.baseline_year_1_arm_1',
    'KSADSintern.baseline_year_1_arm_1',
    'nihtbx_cryst_agecorrected.baseline_year_1_arm_1', 
    'ACEs.baseline_year_1_arm_1',
    'avgPFCthick_QA.baseline_year_1_arm_1',
    'rsfmri_c_ngd_cgc_ngd_cgc_QA.baseline_year_1_arm_1',
    'rsfmri_c_ngd_dt_ngd_dt_QA.baseline_year_1_arm_1'
]

features_followup = [
    #'interview_age.2_year_follow_up_y_arm_1',
    'KSADSintern.2_year_follow_up_y_arm_1',
    'nihtbx_cryst_agecorrected.2_year_follow_up_y_arm_1',
    'ACEs.2_year_follow_up_y_arm_1',
    'avgPFCthick_QA.2_year_follow_up_y_arm_1',
    'rsfmri_c_ngd_cgc_ngd_cgc_QA.2_year_follow_up_y_arm_1',
    'rsfmri_c_ngd_dt_ngd_dt_QA.2_year_follow_up_y_arm_1',
]

features_all_time = features_baseline + features_followup

cross_sectional_features = [
    'rel_family_id',
    #'demo_sex_v2',
    #'race_ethnicity',
    'acs_raked_propensity_score',
    'speechdelays',
    'motordelays',
    'fam_history_8_yes_no',
]

for col in features_all_time + cross_sectional_features:
    df[col] = df[col].astype(str).str.strip()
    df.loc[df[col] == '', col] = np.nan
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 2: Drop rows with missing values in key column(s)
df.dropna(subset=features_all_time + ['group_PDvLP_3timepoint'], inplace=True)

# Step 3: KNN Imputation on features (assumes all are numeric now)
imputer = KNNImputer(n_neighbors=5) #comment these two lines for non data imputation
df[features_all_time + cross_sectional_features] = imputer.fit_transform(df[features_all_time + cross_sectional_features])

print(f"Total samples after cleaning: {len(df)}")

# ===============================
# Leave-One-Site-Out Cross-Validation
# ===============================
site_ids = df['site'].unique()
site_metrics = []
all_cross_importances = []
all_long_importances = []
all_predictions = []



def build_model(input_shape_ts, input_shape_cross, head_size, num_heads, ff_dim, dropout, dense_unit, learning_rate, alpha):
    input_ts = Input(shape=input_shape_ts, name='time_series_input')
    x = transformer_encoder(input_ts, head_size, num_heads, ff_dim, dropout)
    x = Dropout(0.3)(x)
    x = Dense(dense_unit, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    ts_output = x[:, -1, :]  # last timestep

    input_cross = Input(shape=(input_shape_cross,), name='cross_sectional_input')
    concatenated = Concatenate()([ts_output, input_cross])
    x = Dense(32, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(x)


    model = Model(inputs=[input_ts, input_cross], outputs=output)
    model.compile(
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=focal_loss(alpha=alpha),
        metrics=['accuracy']
    )
    return model

head_sizes = [32, 64]
num_heads_list = [2, 4]
ff_dims = [64, 128]
dropouts = [0.1, 0.2]
dense_units = [32, 64]
learning_rates = [1e-3, 3e-4]
alphas = [0.5, 0.75]
results = []


csv_file_path = 'transformerHPResults_remove_demos.csv'

for test_site in site_ids:
    print(f"\n==== Testing on site {test_site} ====")

    # 1. Leave-one-site-out split
    df_train = df[df['site'] != test_site]
    df_test = df[df['site'] == test_site]

    if df_train.empty or df_test.empty:
        print("Skipping due to empty split.")
        continue


    df_majority = df_train[df_train['group_PDvLP_3timepoint'] == 0]
    df_minority = df_train[df_train['group_PDvLP_3timepoint'] == 1]

    # Downsample majority class to size of minority class
    # Upsample minority class to match majority class
    df_minority_oversampled = resample(
        df_minority,
        replace=True,    # sample with replacement
        n_samples=len(df_majority),  # match majority class count
        random_state=42  # reproducible
    )

    # Combine majority class with oversampled minority class
    #df_train = pd.concat([df_majority, df_minority_oversampled])


    if df_train.empty or df_test.empty:
        print(f"Skipping site {test_site} due to empty split.")
        continue

    # Time series sequences
    X_train_ts, y_train = [], []
    for _, row in df_train.iterrows():
        baseline = row[features_baseline].values.astype(np.float32)
        followup = row[features_followup].values.astype(np.float32)
        if baseline.shape != followup.shape:
            continue
        seq = np.stack([baseline, followup])
        X_train_ts.append(seq)
        y_train.append(row['group_PDvLP_3timepoint'])

    X_test_ts, y_test = [], []
    for _, row in df_test.iterrows():
        baseline = row[features_baseline].values.astype(np.float32)
        followup = row[features_followup].values.astype(np.float32)
        if baseline.shape != followup.shape:
            continue
        seq = np.stack([baseline, followup])
        X_test_ts.append(seq)
        y_test.append(row['group_PDvLP_3timepoint'])

    X_train_ts = np.array(X_train_ts)
    X_test_ts = np.array(X_test_ts)
    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)

    if len(X_train_ts) == 0 or len(X_test_ts) == 0:
        print(f"Skipping site {test_site} due to no valid samples.")
        continue

    # Cross-sectional
    X_train_cross = df_train[cross_sectional_features].copy()
    X_test_cross = df_test[cross_sectional_features].copy()
    X_train_cross.fillna(X_train_cross.median(), inplace=True)
    X_test_cross.fillna(X_train_cross.median(), inplace=True)

    # ==================
    # Standardize
    # ==================
    scaler_ts = StandardScaler()
    X_train_ts_flat = X_train_ts.reshape(-1, X_train_ts.shape[2])
    X_test_ts_flat = X_test_ts.reshape(-1, X_test_ts.shape[2])

    X_train_ts_scaled = scaler_ts.fit_transform(X_train_ts_flat).reshape(X_train_ts.shape)
    X_test_ts_scaled = scaler_ts.transform(X_test_ts_flat).reshape(X_test_ts.shape)

    scaler_cross = StandardScaler()
    X_train_cross_scaled = scaler_cross.fit_transform(X_train_cross)
    X_test_cross_scaled = scaler_cross.transform(X_test_cross)


    X_ts_train_final, X_ts_val, X_cross_train_final, X_cross_val, y_train_final, y_val = train_test_split(
        X_train_ts_scaled, X_train_cross_scaled, y_train, test_size=0.1, stratify=y_train, random_state=42
    )

    best_auc = -1
    best_model = None
    best_params = None


    # 4. Grid search
    for alpha in alphas:
        for head_size in head_sizes:
            for num_heads in num_heads_list:
                for ff_dim in ff_dims:
                    for dropout in dropouts:
                        for dense_unit in dense_units:
                            for lr in learning_rates:
                                if os.path.exists(csv_file_path):
                                    existing_results = pd.read_csv(csv_file_path)
                                else:
                                    existing_results = pd.DataFrame()


                                combo = {
                                'site': test_site,
                                'alpha': alpha,
                                'head_size': head_size,
                                'num_heads': num_heads,
                                'ff_dim': ff_dim,
                                'dropout': dropout,
                                'dense_unit': dense_unit,
                                'learning_rate': lr,
                                }
                            # Check if all keys & values exist in existing_results
                                duplicate = False
                                if not existing_results.empty:
                                    # Filter rows matching all combo values exactly
                                    matches = existing_results[
                                        (existing_results['site'] == combo['site']) &
                                        (existing_results['alpha'] == combo['alpha']) &
                                        (existing_results['head_size'] == combo['head_size']) &
                                        (existing_results['num_heads'] == combo['num_heads']) &
                                        (existing_results['ff_dim'] == combo['ff_dim']) &
                                        (existing_results['dropout'] == combo['dropout']) &
                                        (existing_results['dense_unit'] == combo['dense_unit']) &
                                        (existing_results['learning_rate'] == combo['learning_rate'])
                                    ]
                                    if len(matches) > 0:
                                        duplicate = True

                                if duplicate:
                                    print(f"Skipping duplicate combo on site {test_site}: {combo}")
                                    continue

                                print(f"Training on site {test_site} with params: {combo}")

                                model = build_model(
                                    input_shape_ts=(X_ts_train_final.shape[1], X_ts_train_final.shape[2]),
                                    input_shape_cross=X_cross_train_final.shape[1],
                                    head_size=head_size,
                                    num_heads=num_heads,
                                    ff_dim=ff_dim,
                                    dropout=dropout,
                                    dense_unit=dense_unit,
                                    learning_rate=lr,
                                    alpha=alpha
                                )


                                class_weights = compute_class_weight(
                                    class_weight='balanced',
                                    classes=np.unique(y_train_final),
                                    y=y_train_final
                                )
                                class_weights = dict(enumerate(class_weights))

                                model.fit(
                                    [X_ts_train_final, X_cross_train_final], y_train_final,
                                    validation_data=([X_ts_val, X_cross_val], y_val),
                                    epochs=20,
                                    batch_size=16,
                                    verbose=0,
                                    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
                                    #,class_weight=class_weights
                                )

                                val_probs = model.predict([X_ts_val, X_cross_val]).flatten()
                                val_auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) == 2 else float('nan')




                                # Example: inside your grid search loop after model.predict and val_auc calculation

                                val_preds = (val_probs >= 0.5).astype(int)

                                acc = accuracy_score(y_val, val_preds)
                                precision = precision_score(y_val, val_preds)
                                recall = recall_score(y_val, val_preds)
                                f1 = f1_score(y_val, val_preds)

                                tn, fp, fn, tp = confusion_matrix(y_val, val_preds).ravel()
                                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                                npv = tn / (tn + fn) if (tn + fn) > 0 else 0

                                result_row = {
                                'site': test_site,
                                'alpha': alpha,
                                'head_size': head_size,
                                'num_heads': num_heads,
                                'ff_dim': ff_dim,
                                'dropout': dropout,
                                'dense_unit': dense_unit,
                                'learning_rate': lr,
                                'val_auc': val_auc,
                                'accuracy': acc,
                                'precision': precision,
                                'recall': recall,
                                'npv': npv,
                                'specificity': specificity,
                                'f1': f1
                                }
                                existing_results = pd.concat([existing_results, pd.DataFrame([result_row])], ignore_index=True)

                                header = not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0
                                pd.DataFrame([result_row]).to_csv(csv_file_path, mode='a', header=header, index=False)


                                if val_auc > best_auc:
                                    best_auc = val_auc
                                    best_model = model
                                    best_params = {
                                        'head_size': head_size,
                                        'num_heads': num_heads,
                                        'ff_dim': ff_dim,
                                        'dropout': dropout,
                                        'dense_unit': dense_unit,
                                        'lr': lr
                                    }


df_results = pd.read_csv(csv_file_path)
df_results = df_results.sort_values(by='val_auc', ascending=False)
df_results.to_csv(csv_file_path, index=False)

print(f"Saved hyperparameter tuning results to {csv_file_path}")