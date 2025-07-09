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

import matplotlib.pyplot as plt


from sklearn.utils import resample, class_weight
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer

# TensorFlow and Keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import keras.backend as K

# SHAP for explainability
import shap




def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        # Use tf.clip_by_value instead of keras.backend.clip
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        
        loss = weight * cross_entropy
        
        return tf.reduce_mean(loss)
    
    return focal_loss_fixed

# ===========================
# Load and preprocess dataset
# ===========================
df = pd.read_csv('for_JS_final_withgroup.csv')

# Extract numeric site ID from 'site_id_l.baseline_year_1_arm_1'
df['site'] = df['site_id_l.baseline_year_1_arm_1'].str.extract(r'(\d+)$').astype(int)

# Define features for each timepoint
features_baseline = [
    'interview_age.baseline_year_1_arm_1',
    'KSADSintern.baseline_year_1_arm_1',
    'nihtbx_cryst_agecorrected.baseline_year_1_arm_1', 
    'ACEs.baseline_year_1_arm_1',
    'avgPFCthick_QA.baseline_year_1_arm_1',
    'rsfmri_c_ngd_cgc_ngd_cgc_QA.baseline_year_1_arm_1',
    'rsfmri_c_ngd_dt_ngd_dt_QA.baseline_year_1_arm_1'
]

features_followup = [
    'interview_age.2_year_follow_up_y_arm_1',
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
    'demo_sex_v2',
    'race_ethnicity',
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

# ===============================
# Leave-One-Site-Out Cross-Validation
# ===============================
site_ids = df['site'].unique()
site_metrics = []
all_cross_importances = []
all_long_importances = []
all_predictions = []


for test_site in site_ids:
    print(f"\n==== Testing on site {test_site} ====")

    df_train = df[df['site'] != test_site]
    df_test = df[df['site'] == test_site]


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

    # Build sequences for training (time series)
    X_train_ts, y_train = [], []
    for _, row in df_train.iterrows():
        baseline = row[features_baseline].values.astype(np.float32)
        followup = row[features_followup].values.astype(np.float32)
        if baseline.shape != followup.shape:
            continue
        seq = np.stack([baseline, followup])
        X_train_ts.append(seq)
        y_train.append(row['group_PDvLP_3timepoint'])

    # Build sequences for testing (time series)
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

    # Build cross-sectional features for train and test
    X_train_cross = df_train[cross_sectional_features].copy()
    X_test_cross = df_test[cross_sectional_features].copy()

    # Fill missing cross-sectional data with median of train set
    X_train_cross.fillna(X_train_cross.median(), inplace=True)
    X_test_cross.fillna(X_train_cross.median(), inplace=True)

    # Compute class weights
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), class_weights_array))
    #print("Class weights:", class_weights)


    print(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # ==================
    # Standardize features
    # ==================
    scaler_ts = StandardScaler()
    X_train_ts_flat = X_train_ts.reshape(-1, X_train_ts.shape[2])
    X_test_ts_flat = X_test_ts.reshape(-1, X_test_ts.shape[2])

    X_train_ts_scaled = scaler_ts.fit_transform(X_train_ts_flat).reshape(X_train_ts.shape)
    X_test_ts_scaled = scaler_ts.transform(X_test_ts_flat).reshape(X_test_ts.shape)

    scaler_cross = StandardScaler()
    X_train_cross_scaled = scaler_cross.fit_transform(X_train_cross)
    X_test_cross_scaled = scaler_cross.transform(X_test_cross)

    # ================
    # Build LSTM + cross-sectional model
    # ================

    input_ts = Input(shape=(X_train_ts_scaled.shape[1], X_train_ts_scaled.shape[2]), name='time_series_input')
    lstm_out = LSTM(64, return_sequences=False)(input_ts)
    lstm_out = Dropout(0.3)(lstm_out)

    input_cross = Input(shape=(X_train_cross_scaled.shape[1],), name='cross_sectional_input')

    concatenated = Concatenate()([lstm_out, input_cross])
    dense1 = Dense(32, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=[input_ts, input_cross], outputs=output)

    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        [X_train_ts_scaled, X_train_cross_scaled], y_train,
        epochs=20,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
        class_weight=class_weights
    )

    # =========
    # Evaluate
    # =========
    print("X_test_ts_scaled shape:", X_test_ts_scaled.shape)
    print("Any NaNs in X_test_ts_scaled?", np.isnan(X_test_ts_scaled).any())
    print("X_test_cross_scaled shape:", X_test_cross_scaled.shape)
    print("Any NaNs in X_test_cross_scaled?", np.isnan(X_test_cross_scaled).any())

    y_pred_probs = model.predict([X_test_ts_scaled, X_test_cross_scaled]).flatten()
    print("Any NaNs in predictions?", np.isnan(y_pred_probs).any())

    y_pred = (y_pred_probs > 0.5).astype(int)

    if np.isnan(y_pred_probs).any():
        print("NaNs detected in prediction probabilities, skipping AUC calculation for this site.")
        auc = float('nan')
    elif len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_pred_probs)
    else:
        auc = float('nan')

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)


    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        print(f"Confusion matrix is not 2x2 for site {test_site}. It is:\n{cm}")
        tn = fp = fn = tp = 0
    
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"AUC:       {auc:.3f}")
    print(f"Negative Predictive Value (NPV): {npv:.2f}")
    print(f"Specificity: {specificity:.3f}")


    X_ts_for_shap = X_test_ts_scaled[:100]  # or all: X_test_ts_scaled
    X_cross_for_shap = X_test_cross_scaled[:100]
    explainer = shap.GradientExplainer(model, [X_ts_for_shap, X_cross_for_shap])

    # Compute SHAP values for test longitudinal data (along with cross-sectional)
    shap_values = explainer.shap_values([X_test_ts_scaled, X_test_cross_scaled])
    shap_long = shap_values[0]  # shape: (samples, 2, 7)
    shap_cross = shap_values[1]  # SHAP values for cross-sectional input (shape: samples x features)


    # Average absolute SHAP values over samples and time steps for each longitudinal feature
    mean_abs_shap_long = np.mean(np.abs(shap_long), axis=(0,1))  # shape: (7,)
    mean_abs_shap_cross = np.mean(np.abs(shap_cross), axis=0)  # Average over samples



    # Store for averaging later across sites
    all_long_importances.append(mean_abs_shap_long)

    
    all_cross_importances.append(mean_abs_shap_cross)
    site_metrics.append((test_site, acc, prec,rec,npv,auc,specificity,f1))


    df_test_results = df_test[['src_subject_id']].copy()
    df_test_results['true_label'] = y_test
    df_test_results['predicted_label'] = y_pred
    df_test_results['predicted_prob'] = y_pred_probs
    df_test_results['site'] = test_site
    all_predictions.append(df_test_results)

all_predictions_df = pd.concat(all_predictions, ignore_index=True)
all_predictions_df.to_csv('LSTM_all_subject_predictions.csv', index=False)


# Get the true and predicted labels from your aggregated DataFrame
y_true = all_predictions_df['true_label']
y_pred = all_predictions_df['predicted_label']

# Compute and display confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (All Sites)')
plt.show()

# Optional: classification report
print(classification_report(y_true, y_pred))


# =======================
# Report overall accuracy & AUC
# =======================
def mean_ci(data, confidence=0.95):
    a = np.array(data)
    n = len(a)
    mean = np.mean(a)
    sem = stats.sem(a)
    h = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean - h, mean + h


def mean_ci_long(data, confidence=0.95):
    data = np.array(data)
    n = len(data)
    mean = np.mean(data).item()  # .item() converts 0-d array to float
    sem = np.std(data, ddof=1) / np.sqrt(n)
    h = sem * t.ppf((1 + confidence) / 2., n-1)
    lower = (mean - h).item()
    upper = (mean + h).item()
    return mean, lower, upper

print("\n===== Final Report =====")
for site, acc, prec, rec, npv, auc, spec, f1 in site_metrics:
    print(f"Site {site}: Accuracy = {acc:.3f}, Precision = {prec:.3f}, Recall = {rec:.3f}, NPV = {npv:.3f}, AUC = {auc:.3f}, Specificity = {spec:.3f}, F1 = {f1:.3f}")

# Proper unpacking (8 elements per tuple)
acc_list = [acc for _, acc, _, _, _, _, _, _ in site_metrics]
prec_list = [prec for _, _, prec, _, _, _, _, _ in site_metrics]
rec_list = [rec for _, _, _, rec, _, _, _, _ in site_metrics]
npv_list = [npv for _, _, _, _, npv, _, _, _ in site_metrics]
auc_list = [auc for _, _, _, _, _, auc, _, _ in site_metrics if not np.isnan(auc)]
spec_list = [spec for _, _, _, _, _, _, spec, _ in site_metrics]
f1_list = [f1 for _, _, _, _, _, _, _, f1 in site_metrics]

avg_acc, acc_lower, acc_upper = mean_ci(acc_list)
avg_prec, prec_lower, prec_upper = mean_ci(prec_list)
avg_rec, rec_lower, rec_upper = mean_ci(rec_list)
avg_npv, npv_lower, npv_upper = mean_ci(npv_list)
avg_auc, auc_lower, auc_upper = mean_ci(auc_list)
avg_spec, spec_lower, spec_upper = mean_ci(spec_list)
avg_f1, f1_lower, f1_upper = mean_ci(f1_list)

print("\n===== LOSO Summary (Average ± 95% CI) =====")
print(f"Accuracy:                {avg_acc:.3f} [{acc_lower:.3f}, {acc_upper:.3f}]")
print(f"Precision (PPV):         {avg_prec:.3f} [{prec_lower:.3f}, {prec_upper:.3f}]")
print(f"Recall (Sensitivity):    {avg_rec:.3f} [{rec_lower:.3f}, {rec_upper:.3f}]")
print(f"NPV:                     {avg_npv:.3f} [{npv_lower:.3f}, {npv_upper:.3f}]")
print(f"AUC:                     {avg_auc:.3f} [{auc_lower:.3f}, {auc_upper:.3f}]")
print(f"Specificity:             {avg_spec:.3f} [{spec_lower:.3f}, {spec_upper:.3f}]")
print(f"F1 Score:                {avg_f1:.3f} [{f1_lower:.3f}, {f1_upper:.3f}]")


all_cross_importances = np.array(all_cross_importances)  # shape: (n_folds, n_cross_features)

print("\n=== Average Cross-Sectional Feature Importances (SHAP) ===")
for i, feat in enumerate(cross_sectional_features):  # cross-sectional feature names
    mean, lower, upper = mean_ci_long(all_cross_importances[:, i])
    print(f"{feat}: {mean:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")

all_long_importances = np.array(all_long_importances)  # shape: (n_folds, n_features)

print("\n=== Average Longitudinal Feature Importances (SHAP) ===")
for i, feat in enumerate(features_baseline):  # use your longitudinal feature names
    mean, lower, upper = mean_ci_long(all_long_importances[:, i])
    print(f"{feat}: {mean:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")




# ===== Final Report =====
# Site 21: Accuracy = 0.867, Precision = 0.241, Recall = 0.500, NPV = 0.963, AUC = 0.767, Specificity = 0.892
# Site 11: Accuracy = 0.764, Precision = 0.083, Recall = 1.000, NPV = 1.000, AUC = 0.998, Specificity = 0.759
# Site 4: Accuracy = 0.719, Precision = 0.089, Recall = 0.500, NPV = 0.965, AUC = 0.683, Specificity = 0.730
# Site 5: Accuracy = 0.900, Precision = 0.231, Recall = 0.600, NPV = 0.981, AUC = 0.748, Specificity = 0.913
# Site 6: Accuracy = 0.856, Precision = 0.079, Recall = 0.600, NPV = 0.991, AUC = 0.760, Specificity = 0.861
# Site 20: Accuracy = 0.772, Precision = 0.220, Recall = 0.769, NPV = 0.976, AUC = 0.852, Specificity = 0.772
# Site 1: Accuracy = 0.709, Precision = 0.125, Recall = 0.667, NPV = 0.972, AUC = 0.753, Specificity = 0.711
# Site 18: Accuracy = 0.823, Precision = 0.208, Recall = 0.833, NPV = 0.989, AUC = 0.947, Specificity = 0.822
# Site 3: Accuracy = 0.744, Precision = 0.025, Recall = 0.250, NPV = 0.976, AUC = 0.677, Specificity = 0.756
# Site 12: Accuracy = 0.861, Precision = 0.222, Recall = 0.750, NPV = 0.986, AUC = 0.814, Specificity = 0.866
# Site 14: Accuracy = 0.934, Precision = 0.200, Recall = 0.800, NPV = 0.996, AUC = 0.933, Specificity = 0.937
# Site 10: Accuracy = 0.831, Precision = 0.294, Recall = 0.625, NPV = 0.958, AUC = 0.807, Specificity = 0.851
# Site 16: Accuracy = 0.943, Precision = 0.200, Recall = 0.500, NPV = 0.988, AUC = 0.938, Specificity = 0.953
# Site 13: Accuracy = 0.751, Precision = 0.088, Recall = 1.000, NPV = 1.000, AUC = 0.941, Specificity = 0.745
# Site 9: Accuracy = 0.784, Precision = 0.171, Recall = 1.000, NPV = 1.000, AUC = 0.921, Specificity = 0.773
# Site 2: Accuracy = 0.877, Precision = 0.040, Recall = 0.250, NPV = 0.985, AUC = 0.735, Specificity = 0.888
# Site 15: Accuracy = 0.699, Precision = 0.048, Recall = 1.000, NPV = 1.000, AUC = 0.779, Specificity = 0.695
# Site 19: Accuracy = 0.785, Precision = 0.111, Recall = 0.714, NPV = 0.987, AUC = 0.843, Specificity = 0.787
# Site 8: Accuracy = 0.806, Precision = 0.059, Recall = 0.333, NPV = 0.974, AUC = 0.593, Specificity = 0.822
# Site 17: Accuracy = 0.923, Precision = 0.100, Recall = 1.000, NPV = 1.000, AUC = 0.987, Specificity = 0.922
# Site 7: Accuracy = 0.882, Precision = 0.111, Recall = 1.000, NPV = 1.000, AUC = 1.000, Specificity = 0.881
# Site 22: Accuracy = 0.875, Precision = 0.000, Recall = 0.000, NPV = 1.000, AUC = nan, Specificity = 0.875

# ===== LOSO Summary (Average ± 95% CI) =====
# Accuracy:                0.823 [0.790, 0.856]
# Precision (PPV):         0.134 [0.098, 0.170]
# Recall (Sensitivity):    0.668 [0.541, 0.795]
# NPV:                     0.986 [0.980, 0.992]
# AUC:                     0.832 [0.779, 0.885]
# Specificity:             0.828 [0.794, 0.862]

# === Average Cross-Sectional Feature Importances (SHAP) ===
# rel_family_id: 0.0017 (95% CI: [0.0014, 0.0021])
# demo_sex_v2: 0.0716 (95% CI: [0.0637, 0.0795])
# race_ethnicity: 0.0282 (95% CI: [0.0216, 0.0347])
# acs_raked_propensity_score: 0.0095 (95% CI: [0.0077, 0.0114])
# speechdelays: 0.0110 (95% CI: [0.0075, 0.0145])
# motordelays: 0.0101 (95% CI: [0.0082, 0.0120])
# fam_history_8_yes_no: 0.0102 (95% CI: [0.0081, 0.0124])

# === Average Longitudinal Feature Importances (SHAP) ===
# interview_age.baseline_year_1_arm_1: 0.0145 (95% CI: [0.0125, 0.0165])
# KSADSintern.baseline_year_1_arm_1: 0.0435 (95% CI: [0.0382, 0.0488])
# nihtbx_cryst_agecorrected.baseline_year_1_arm_1: 0.0287 (95% CI: [0.0246, 0.0327])
# ACEs.baseline_year_1_arm_1: 0.0428 (95% CI: [0.0396, 0.0460])
# avgPFCthick_QA.baseline_year_1_arm_1: 0.0148 (95% CI: [0.0122, 0.0175])
# rsfmri_c_ngd_cgc_ngd_cgc_QA.baseline_year_1_arm_1: 0.0117 (95% CI: [0.0089, 0.0145])
# rsfmri_c_ngd_dt_ngd_dt_QA.baseline_year_1_arm_1: 0.0166 (95% CI: [0.0143, 0.0190])




# LSTM Minority Class Oversampling
# ===== Final Report =====
# Site 21: Accuracy = 0.927, Precision = 0.417, Recall = 0.357, NPV = 0.956, AUC = 0.761, Specificity = 0.966, F1 = 0.385
# Site 11: Accuracy = 0.979, Precision = 0.500, Recall = 0.667, NPV = 0.993, AUC = 0.981, Specificity = 0.985, F1 = 0.571
# Site 4: Accuracy = 0.893, Precision = 0.167, Recall = 0.286, NPV = 0.961, AUC = 0.666, Specificity = 0.925, F1 = 0.211
# Site 5: Accuracy = 0.942, Precision = 0.250, Recall = 0.200, NPV = 0.966, AUC = 0.725, Specificity = 0.974, F1 = 0.222
# Site 6: Accuracy = 0.946, Precision = 0.091, Recall = 0.200, NPV = 0.984, AUC = 0.731, Specificity = 0.960, F1 = 0.125
# Site 20: Accuracy = 0.902, Precision = 0.360, Recall = 0.346, NPV = 0.946, AUC = 0.784, Specificity = 0.949, F1 = 0.353
# Site 1: Accuracy = 0.864, Precision = 0.100, Recall = 0.167, NPV = 0.946, AUC = 0.624, Specificity = 0.907, F1 = 0.125
# Site 18: Accuracy = 0.956, Precision = 0.600, Recall = 0.500, NPV = 0.972, AUC = 0.883, Specificity = 0.981, F1 = 0.545
# Site 3: Accuracy = 0.921, Precision = 0.091, Recall = 0.250, NPV = 0.980, AUC = 0.848, Specificity = 0.938, F1 = 0.133
# Site 12: Accuracy = 0.939, Precision = 0.333, Recall = 0.250, NPV = 0.962, AUC = 0.795, Specificity = 0.975, F1 = 0.286
# Site 14: Accuracy = 0.965, Precision = 0.167, Recall = 0.200, NPV = 0.984, AUC = 0.763, Specificity = 0.980, F1 = 0.182
# Site 10: Accuracy = 0.917, Precision = 0.600, Recall = 0.250, NPV = 0.930, AUC = 0.683, Specificity = 0.983, F1 = 0.353
# Site 16: Accuracy = 0.970, Precision = 0.364, Recall = 0.400, NPV = 0.986, AUC = 0.847, Specificity = 0.984, F1 = 0.381
# Site 13: Accuracy = 0.928, Precision = 0.188, Recall = 0.600, NPV = 0.990, AUC = 0.875, Specificity = 0.936, F1 = 0.286
# Site 9: Accuracy = 0.955, Precision = 0.500, Recall = 0.500, NPV = 0.977, AUC = 0.697, Specificity = 0.977, F1 = 0.500
# Site 2: Accuracy = 0.959, Precision = 0.143, Recall = 0.250, NPV = 0.986, AUC = 0.634, Specificity = 0.972, F1 = 0.182
# Site 15: Accuracy = 0.917, Precision = 0.000, Recall = 0.000, NPV = 0.984, AUC = 0.397, Specificity = 0.931, F1 = 0.000
# Site 19: Accuracy = 0.933, Precision = 0.250, Recall = 0.429, NPV = 0.978, AUC = 0.796, Specificity = 0.952, F1 = 0.316
# Site 8: Accuracy = 0.968, Precision = 0.500, Recall = 0.333, NPV = 0.978, AUC = 0.470, Specificity = 0.989, F1 = 0.400
# Site 17: Accuracy = 0.974, Precision = 0.167, Recall = 0.500, NPV = 0.996, AUC = 0.970, Specificity = 0.978, F1 = 0.250
# Site 7: Accuracy = 0.971, Precision = 0.333, Recall = 1.000, NPV = 1.000, AUC = 1.000, Specificity = 0.970, F1 = 0.500
# Site 22: Accuracy = 0.875, Precision = 0.000, Recall = 0.000, NPV = 1.000, AUC = nan, Specificity = 0.875, F1 = 0.000

# ===== LOSO Summary (Average ± 95% CI) =====
# Accuracy:                0.936 [0.922, 0.951]
# Precision (PPV):         0.278 [0.196, 0.360]
# Recall (Sensitivity):    0.349 [0.250, 0.449]
# NPV:                     0.975 [0.967, 0.983]
# AUC:                     0.759 [0.689, 0.828]
# Specificity:             0.959 [0.946, 0.972]
# F1 Score:                0.287 [0.215, 0.359]

# === Average Cross-Sectional Feature Importances (SHAP) ===
# rel_family_id: 0.0014 (95% CI: [0.0010, 0.0019])
# demo_sex_v2: 0.0308 (95% CI: [0.0239, 0.0378])
# race_ethnicity: 0.0152 (95% CI: [0.0113, 0.0192])
# acs_raked_propensity_score: 0.0117 (95% CI: [0.0095, 0.0139])
# speechdelays: 0.0110 (95% CI: [0.0080, 0.0140])
# motordelays: 0.0131 (95% CI: [0.0102, 0.0161])
# fam_history_8_yes_no: 0.0125 (95% CI: [0.0051, 0.0199])

# === Average Longitudinal Feature Importances (SHAP) ===
# interview_age.baseline_year_1_arm_1: 0.0132 (95% CI: [0.0101, 0.0164])
# KSADSintern.baseline_year_1_arm_1: 0.0306 (95% CI: [0.0233, 0.0380])
# nihtbx_cryst_agecorrected.baseline_year_1_arm_1: 0.0182 (95% CI: [0.0140, 0.0223])
# ACEs.baseline_year_1_arm_1: 0.0289 (95% CI: [0.0230, 0.0349])
# avgPFCthick_QA.baseline_year_1_arm_1: 0.0207 (95% CI: [0.0160, 0.0254])
# rsfmri_c_ngd_cgc_ngd_cgc_QA.baseline_year_1_arm_1: 0.0207 (95% CI: [0.0177, 0.0237])
# rsfmri_c_ngd_dt_ngd_dt_QA.baseline_year_1_arm_1: 0.0198 (95% CI: [0.0167, 0.0230])



#data imputation - minority over sampling 

# ===== Final Report =====
# Site 21: Accuracy = 0.922, Precision = 0.385, Recall = 0.357, NPV = 0.956, AUC = 0.749, Specificity = 0.961, F1 = 0.370
# Site 11: Accuracy = 0.943, Precision = 0.273, Recall = 1.000, NPV = 1.000, AUC = 0.959, Specificity = 0.942, F1 = 0.429
# Site 4: Accuracy = 0.900, Precision = 0.182, Recall = 0.286, NPV = 0.961, AUC = 0.706, Specificity = 0.933, F1 = 0.222
# Site 5: Accuracy = 0.942, Precision = 0.250, Recall = 0.200, NPV = 0.966, AUC = 0.736, Specificity = 0.974, F1 = 0.222
# Site 6: Accuracy = 0.957, Precision = 0.200, Recall = 0.400, NPV = 0.988, AUC = 0.695, Specificity = 0.968, F1 = 0.267
# Site 20: Accuracy = 0.905, Precision = 0.286, Recall = 0.154, NPV = 0.932, AUC = 0.771, Specificity = 0.968, F1 = 0.200
# Site 1: Accuracy = 0.893, Precision = 0.273, Recall = 0.500, NPV = 0.967, AUC = 0.698, Specificity = 0.918, F1 = 0.353
# Site 18: Accuracy = 0.956, Precision = 0.667, Recall = 0.333, NPV = 0.964, AUC = 0.877, Specificity = 0.991, F1 = 0.444
# Site 3: Accuracy = 0.902, Precision = 0.000, Recall = 0.000, NPV = 0.974, AUC = 0.834, Specificity = 0.925, F1 = 0.000
# Site 12: Accuracy = 0.933, Precision = 0.200, Recall = 0.125, NPV = 0.956, AUC = 0.766, Specificity = 0.975, F1 = 0.154
# Site 14: Accuracy = 0.965, Precision = 0.250, Recall = 0.400, NPV = 0.988, AUC = 0.754, Specificity = 0.976, F1 = 0.308
# Site 10: Accuracy = 0.895, Precision = 0.333, Recall = 0.167, NPV = 0.921, AUC = 0.712, Specificity = 0.967, F1 = 0.222
# Site 16: Accuracy = 0.957, Precision = 0.200, Recall = 0.300, NPV = 0.983, AUC = 0.834, Specificity = 0.972, F1 = 0.240
# Site 13: Accuracy = 0.928, Precision = 0.222, Recall = 0.800, NPV = 0.995, AUC = 0.892, Specificity = 0.931, F1 = 0.348
# Site 9: Accuracy = 0.933, Precision = 0.286, Recall = 0.333, NPV = 0.969, AUC = 0.734, Specificity = 0.961, F1 = 0.308
# Site 2: Accuracy = 0.950, Precision = 0.182, Recall = 0.500, NPV = 0.990, AUC = 0.652, Specificity = 0.958, F1 = 0.267
# Site 15: Accuracy = 0.917, Precision = 0.000, Recall = 0.000, NPV = 0.984, AUC = 0.561, Specificity = 0.931, F1 = 0.000
# Site 19: Accuracy = 0.938, Precision = 0.273, Recall = 0.429, NPV = 0.978, AUC = 0.798, Specificity = 0.957, F1 = 0.333
# Site 8: Accuracy = 0.935, Precision = 0.200, Recall = 0.333, NPV = 0.977, AUC = 0.463, Specificity = 0.956, F1 = 0.250
# Site 17: Accuracy = 0.949, Precision = 0.083, Recall = 0.500, NPV = 0.995, AUC = 0.957, Specificity = 0.953, F1 = 0.143
# Site 7: Accuracy = 0.971, Precision = 0.333, Recall = 1.000, NPV = 1.000, AUC = 1.000, Specificity = 0.970, F1 = 0.500
# Site 22: Accuracy = 0.875, Precision = 0.000, Recall = 0.000, NPV = 1.000, AUC = nan, Specificity = 0.875, F1 = 0.000

# ===== LOSO Summary (Average ± 95% CI) =====
# Accuracy:                0.930 [0.919, 0.942]
# Precision (PPV):         0.231 [0.167, 0.295]
# Recall (Sensitivity):    0.369 [0.245, 0.493]
# NPV:                     0.975 [0.965, 0.984]
# AUC:                     0.769 [0.710, 0.828]
# Specificity:             0.953 [0.941, 0.964]
# F1 Score:                0.254 [0.193, 0.314]

# === Average Cross-Sectional Feature Importances (SHAP) ===
# rel_family_id: 0.0017 (95% CI: [0.0011, 0.0023])
# demo_sex_v2: 0.0376 (95% CI: [0.0277, 0.0475])
# race_ethnicity: 0.0190 (95% CI: [0.0138, 0.0242])
# acs_raked_propensity_score: 0.0129 (95% CI: [0.0104, 0.0153])
# speechdelays: 0.0111 (95% CI: [0.0086, 0.0136])
# motordelays: 0.0141 (95% CI: [0.0114, 0.0168])
# fam_history_8_yes_no: 0.0125 (95% CI: [0.0078, 0.0173])

# === Average Longitudinal Feature Importances (SHAP) ===
# interview_age.baseline_year_1_arm_1: 0.0146 (95% CI: [0.0116, 0.0176])
# KSADSintern.baseline_year_1_arm_1: 0.0301 (95% CI: [0.0245, 0.0356])
# nihtbx_cryst_agecorrected.baseline_year_1_arm_1: 0.0207 (95% CI: [0.0164, 0.0250])
# ACEs.baseline_year_1_arm_1: 0.0317 (95% CI: [0.0262, 0.0373])
# avgPFCthick_QA.baseline_year_1_arm_1: 0.0218 (95% CI: [0.0171, 0.0265])
# rsfmri_c_ngd_cgc_ngd_cgc_QA.baseline_year_1_arm_1: 0.0240 (95% CI: [0.0208, 0.0273])
# rsfmri_c_ngd_dt_ngd_dt_QA.baseline_year_1_arm_1: 0.0227 (95% CI: [0.0186, 0.0268])
