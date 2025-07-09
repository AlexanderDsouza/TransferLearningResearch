# Core libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy

# SciPy for statistical analysis
from scipy import stats
from scipy.stats import t

# Scikit-learn utilities
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, balanced_accuracy_score, ConfusionMatrixDisplay, classification_report
)
from sklearn.metrics import precision_recall_curve


import matplotlib.pyplot as plt


from sklearn.utils import resample, class_weight
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


# TensorFlow and Keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import keras.backend as K

# SHAP for explainability
import shap


def focal_loss(gamma=2.0, alpha=0.75):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        loss = -alpha_t * tf.pow(1. - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)
    return loss




# ===========================
# Load and preprocess dataset
# ===========================

df = pd.read_csv('for_JS_final_withgroup.csv')



# Extract numeric site ID from 'site_id_l.baseline_year_1_arm_1'
df['site'] = df['site_id_l.baseline_year_1_arm_1'].str.extract(r'(\d+)$').astype(int)

# Define features for each timepoint
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

# Define cross-sectional features: all other numeric columns not in features_all_time or site/group columns
cross_sectional_features = [
    'rel_family_id',
    #'demo_sex_v2',
    #'race_ethnicity',
    'acs_raked_propensity_score',
    'speechdelays',
    'motordelays',
    'fam_history_8_yes_no',
    ]
# ========================
# Clean data
# ========================



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
print(f"1s: {len(df[df['group_PDvLP_3timepoint'] == 1])} 0s: {len(df[df['group_PDvLP_3timepoint'] == 0])}")
site_ids = df['site'].unique()


def train_loso_with_tuning(
    df, site_ids, 
    features_baseline, features_followup, cross_sectional_features,
    focal_loss,
    lstm_units_list=[32, 64], dropout_rates=[0.2, 0.3],
    dense_units_list=[16, 32], batch_sizes=[16, 32]
):
    results = []

    for test_site in site_ids:
        print(f"\n==== Testing on site {test_site} ====")

        # Split train/test by site
        df_train = df[df['site'] != test_site]
        df_test = df[df['site'] == test_site]

        # Upsample minority if you want (comment/uncomment)
        df_majority = df_train[df_train['group_PDvLP_3timepoint'] == 0]
        df_minority = df_train[df_train['group_PDvLP_3timepoint'] == 1]
        df_minority_oversampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )
        # Comment/uncomment below to toggle oversampling
        #df_train = pd.concat([df_majority, df_minority_oversampled])

        # Build sequences for train and test
        def build_sequences(df_subset):
            X_ts, y = [], []
            for _, row in df_subset.iterrows():
                baseline = row[features_baseline].values.astype(np.float32)
                followup = row[features_followup].values.astype(np.float32)
                if baseline.shape != followup.shape:
                    continue
                seq = np.stack([baseline, followup])
                X_ts.append(seq)
                y.append(row['group_PDvLP_3timepoint'])
            return np.array(X_ts), np.array(y).astype(int)

        X_train_ts, y_train = build_sequences(df_train)
        X_test_ts, y_test = build_sequences(df_test)

        if len(X_train_ts) == 0 or len(X_test_ts) == 0:
            print(f"Skipping site {test_site} due to no valid samples.")
            continue

        # Cross-sectional features
        X_train_cross = df_train[cross_sectional_features].copy()
        X_test_cross = df_test[cross_sectional_features].copy()
        X_train_cross.fillna(X_train_cross.median(), inplace=True)
        X_test_cross.fillna(X_train_cross.median(), inplace=True)

        # Train/val split within train set
        X_train_ts_final, X_val_ts, X_train_cross_final, X_val_cross, y_train_final, y_val = train_test_split(
            X_train_ts, X_train_cross, y_train,
            test_size=0.2, random_state=42, stratify=y_train
        )

        # Compute class weights for balanced training
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_final),
            y=y_train_final
        )
        class_weights = dict(zip(np.unique(y_train_final), class_weights_array))

        # Scale time series data
        scaler_ts = StandardScaler()
        X_train_ts_flat = X_train_ts_final.reshape(-1, X_train_ts_final.shape[2])
        X_val_ts_flat = X_val_ts.reshape(-1, X_val_ts.shape[2])
        X_test_ts_flat = X_test_ts.reshape(-1, X_test_ts.shape[2])

        X_train_ts_scaled = scaler_ts.fit_transform(X_train_ts_flat).reshape(X_train_ts_final.shape)
        X_val_ts_scaled = scaler_ts.transform(X_val_ts_flat).reshape(X_val_ts.shape)
        X_test_ts_scaled = scaler_ts.transform(X_test_ts_flat).reshape(X_test_ts.shape)

        # Scale cross-sectional data
        scaler_cross = StandardScaler()
        X_train_cross_scaled = scaler_cross.fit_transform(X_train_cross_final)
        X_val_cross_scaled = scaler_cross.transform(X_val_cross)
        X_test_cross_scaled = scaler_cross.transform(X_test_cross)

        # Loop over hyperparameters
        alphas = [0.75, 0.25, 0.5]
        gammas = [2.0]

        for alpha in alphas:
            for gamma in gammas:
                for lstm_units in lstm_units_list:
                    for dropout_rate in dropout_rates:
                        for dense_units in dense_units_list:
                            for batch_size in batch_sizes:

                                # Build model
                                input_ts = Input(shape=(X_train_ts_scaled.shape[1], X_train_ts_scaled.shape[2]), name='time_series_input')
                                lstm_out = LSTM(lstm_units, return_sequences=False)(input_ts)
                                lstm_out = Dropout(dropout_rate)(lstm_out)

                                input_cross = Input(shape=(X_train_cross_scaled.shape[1],), name='cross_sectional_input')

                                concatenated = Concatenate()([lstm_out, input_cross])
                                dense1 = Dense(dense_units, activation='relu')(concatenated)
                                output = Dense(1, activation='sigmoid')(dense1)

                                model = Model(inputs=[input_ts, input_cross], outputs=output)
                                model.compile(optimizer='adam', loss=focal_loss(alpha=alpha, gamma=gamma), metrics=['accuracy'])

                                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

                                history = model.fit(
                                    [X_train_ts_scaled, X_train_cross_scaled], y_train_final,
                                    validation_data=([X_val_ts_scaled, X_val_cross_scaled], y_val),
                                    epochs=20,
                                    batch_size=batch_size,
                                    callbacks=[early_stop],
                                    verbose=0,
                                    class_weight=class_weights
                                )

                                val_preds = model.predict([X_val_ts_scaled, X_val_cross_scaled], verbose=0).flatten()
                                val_auc = roc_auc_score(y_val, val_preds)
                                val_preds_binary = (val_preds > 0.5).astype(int)

                                # Compute other metrics
                                acc = accuracy_score(y_val, val_preds_binary)
                                precision = precision_score(y_val, val_preds_binary, zero_division=0)
                                recall = recall_score(y_val, val_preds_binary, zero_division=0)
                                f1 = f1_score(y_val, val_preds_binary, zero_division=0)
                                cm = confusion_matrix(y_val, val_preds_binary)

                                # Extract TN, FP, FN, TP from confusion matrix
                                tn, fp, fn, tp = cm.ravel()

                                specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
                                npv = tn / (tn + fn) if (tn + fn) != 0 else 0.0

                                print(f"Site {test_site}: alpha={alpha}, gamma={gamma}, "
                                    f"LSTM={lstm_units}, Dropout={dropout_rate}, Dense={dense_units}, Batch={batch_size} -> "
                                    f"Val AUC: {val_auc:.4f}, Acc: {acc:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}, "
                                    f"NPV: {npv:.3f}, Spec: {specificity:.3f}, F1: {f1:.3f}")
                                

                                results.append({
                                    'site': test_site,
                                    'alpha': alpha,
                                    'gamma': gamma,
                                    'lstm_units': lstm_units,
                                    'dropout_rate': dropout_rate,
                                    'dense_units': dense_units,
                                    'batch_size': batch_size,
                                    'val_auc': val_auc,
                                    'acc': acc,
                                    'precision': precision,
                                    'recall': recall,
                                    'npv': npv,
                                    'specificity': specificity,
                                    'f1': f1
                                    })
                                


    return results


results = train_loso_with_tuning(
    df, site_ids, 
    features_baseline, features_followup, cross_sectional_features,
    focal_loss=focal_loss,  # your focal loss function
    lstm_units_list=[32, 64],
    dropout_rates=[0.2, 0.3],
    dense_units_list=[16, 32],
    batch_sizes=[16, 32]
)

# Sort results by val_auc descending
results_sorted = sorted(results, key=lambda x: x['val_auc'], reverse=True)

print("\nBest hyperparameter combos:")
for r in results_sorted[:5]:
    print(r)


df_results = pd.DataFrame(results_sorted)
df_results.to_csv('LSTMhyperparameters_exlclude_demos.csv', index=False)