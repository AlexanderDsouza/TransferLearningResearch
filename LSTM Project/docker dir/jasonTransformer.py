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
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Concatenate,
    LayerNormalization, MultiHeadAttention, Add
)
import keras.backend as K

from tensorflow.keras.callbacks import EarlyStopping
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

df['site'] = df['site_id_l.baseline_year_1_arm_1'].str.extract(r'(\d+)$').astype(int)

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

cross_sectional_features = [
    'rel_family_id',
    'demo_sex_v2',
    'race_ethnicity',
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


# === Define Transformer block ===
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



def permutation_importance(model, X_ts, X_cross, y_true, cross_features, metric_func, baseline_score):
    importances = {}

    for i, feature in enumerate(cross_features):
        X_cross_shuffled = deepcopy(X_cross)
        np.random.shuffle(X_cross_shuffled[:, i])
        y_pred_probs = model.predict([X_ts, X_cross_shuffled]).flatten()

        if np.isnan(y_pred_probs).any():
            importance = 0
        else:
            # Convert probabilities to binary predictions for metrics like accuracy
            y_pred_classes = (y_pred_probs >= 0.5).astype(int)
            new_score = metric_func(y_true, y_pred_classes)
            importance = baseline_score - new_score

        importances[feature] = importance

    return importances



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

    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), class_weights_array))

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

    # ================
    # Build Transformer + Cross-Sectional model
    # ================
    input_ts = Input(shape=(X_train_ts_scaled.shape[1], X_train_ts_scaled.shape[2]), name='time_series_input')
    x = transformer_encoder(input_ts)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', name='ts_dense1')(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    ts_output = x[:, -1, :]  # Take last time step

    input_cross = Input(shape=(X_train_cross_scaled.shape[1],), name='cross_sectional_input')
    concatenated = Concatenate()([ts_output, input_cross])
    dense1 = Dense(32, activation='relu', name='combined_dense1')(concatenated)
    output = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=[input_ts, input_cross], outputs=output)
    
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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

    y_pred_probs = model.predict([X_test_ts_scaled, X_test_cross_scaled]).flatten()
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



    # ======================
    # Feature Importance
    # ======================
    # 1) Cross-sectional features importance via permutation importance on accuracy
    baseline_acc = acc
    cross_importances = permutation_importance(
        model, X_test_ts_scaled, X_test_cross_scaled, y_test,
        cross_sectional_features, accuracy_score, baseline_acc
    )

    # Sort by importance (descending)
    cross_importances_sorted = dict(sorted(cross_importances.items(), key=lambda item: item[1], reverse=True))

    # print("\nCross-sectional Feature Importances (by accuracy drop on shuffle):")
    # for feat, imp in cross_importances_sorted.items():
    #     print(f"{feat}: {imp:.4f}")


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
    site_metrics.append((test_site, acc, prec,rec,npv,auc,specificity,f1))
    all_cross_importances.append(mean_abs_shap_cross)


    df_test_results = df_test[['src_subject_id']].copy()
    df_test_results['true_label'] = y_test
    df_test_results['predicted_label'] = y_pred
    df_test_results['predicted_prob'] = y_pred_probs
    df_test_results['site'] = test_site
    all_predictions.append(df_test_results)

    


all_predictions_df = pd.concat(all_predictions, ignore_index=True)
all_predictions_df.to_csv('transformer_all_subject_predictions.csv', index=False)

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
# Final Report
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
# Site 21: Accuracy = 0.761, Precision = 0.148, Recall = 0.571, NPV = 0.963, AUC = 0.749, Specificity = 0.775
# Site 11: Accuracy = 0.721, Precision = 0.071, Recall = 1.000, NPV = 1.000, AUC = 0.998, Specificity = 0.715
# Site 4: Accuracy = 0.473, Precision = 0.059, Recall = 0.643, NPV = 0.961, AUC = 0.656, Specificity = 0.464
# Site 5: Accuracy = 0.900, Precision = 0.231, Recall = 0.600, NPV = 0.981, AUC = 0.748, Specificity = 0.913
# Site 6: Accuracy = 0.840, Precision = 0.071, Recall = 0.600, NPV = 0.991, AUC = 0.735, Specificity = 0.845
# Site 20: Accuracy = 0.813, Precision = 0.253, Recall = 0.731, NPV = 0.973, AUC = 0.864, Specificity = 0.820
# Site 1: Accuracy = 0.718, Precision = 0.129, Recall = 0.667, NPV = 0.972, AUC = 0.722, Specificity = 0.722
# Site 18: Accuracy = 0.805, Precision = 0.214, Recall = 1.000, NPV = 1.000, AUC = 0.931, Specificity = 0.794
# Site 3: Accuracy = 0.701, Precision = 0.041, Recall = 0.500, NPV = 0.983, AUC = 0.744, Specificity = 0.706
# Site 12: Accuracy = 0.903, Precision = 0.250, Recall = 0.500, NPV = 0.973, AUC = 0.885, Specificity = 0.924
# Site 14: Accuracy = 0.949, Precision = 0.250, Recall = 0.800, NPV = 0.996, AUC = 0.901, Specificity = 0.952
# Site 10: Accuracy = 0.711, Precision = 0.202, Recall = 0.750, NPV = 0.966, AUC = 0.813, Specificity = 0.707
# Site 16: Accuracy = 0.943, Precision = 0.222, Recall = 0.600, NPV = 0.990, AUC = 0.930, Specificity = 0.951
# Site 13: Accuracy = 0.565, Precision = 0.052, Recall = 1.000, NPV = 1.000, AUC = 0.896, Specificity = 0.554
# Site 9: Accuracy = 0.821, Precision = 0.179, Recall = 0.833, NPV = 0.991, AUC = 0.863, Specificity = 0.820
# Site 2: Accuracy = 0.868, Precision = 0.037, Recall = 0.250, NPV = 0.984, AUC = 0.745, Specificity = 0.879
# Site 15: Accuracy = 0.586, Precision = 0.035, Recall = 1.000, NPV = 1.000, AUC = 0.889, Specificity = 0.580
# Site 19: Accuracy = 0.800, Precision = 0.136, Recall = 0.857, NPV = 0.993, AUC = 0.814, Specificity = 0.798
# Site 8: Accuracy = 0.839, Precision = 0.071, Recall = 0.333, NPV = 0.975, AUC = 0.478, Specificity = 0.856
# Site 17: Accuracy = 0.902, Precision = 0.080, Recall = 1.000, NPV = 1.000, AUC = 0.987, Specificity = 0.901
# Site 7: Accuracy = 0.794, Precision = 0.067, Recall = 1.000, NPV = 1.000, AUC = 1.000, Specificity = 0.791
# Site 22: Accuracy = 0.875, Precision = 0.000, Recall = 0.000, NPV = 1.000, AUC = nan, Specificity = 0.875

# ===== LOSO Summary (Average ± 95% CI) =====
# Accuracy:                0.786 [0.731, 0.841]
# Precision (PPV):         0.127 [0.090, 0.164]
# Recall (Sensitivity):    0.693 [0.572, 0.813]
# NPV:                     0.986 [0.980, 0.992]
# AUC:                     0.826 [0.768, 0.884]
# Specificity:             0.788 [0.731, 0.846]

# === Average Cross-Sectional Feature Importances (SHAP) ===
# rel_family_id: 0.0015 (95% CI: [0.0011, 0.0019])
# demo_sex_v2: 0.0461 (95% CI: [0.0359, 0.0562])
# race_ethnicity: 0.0171 (95% CI: [0.0131, 0.0210])
# acs_raked_propensity_score: 0.0095 (95% CI: [0.0075, 0.0115])
# speechdelays: 0.0095 (95% CI: [0.0062, 0.0128])
# motordelays: 0.0086 (95% CI: [0.0065, 0.0107])
# fam_history_8_yes_no: 0.0086 (95% CI: [0.0062, 0.0111])

# === Average Longitudinal Feature Importances (SHAP) ===
# interview_age.baseline_year_1_arm_1: 0.0177 (95% CI: [0.0149, 0.0205])
# KSADSintern.baseline_year_1_arm_1: 0.0348 (95% CI: [0.0300, 0.0397])
# nihtbx_cryst_agecorrected.baseline_year_1_arm_1: 0.0327 (95% CI: [0.0274, 0.0380])
# ACEs.baseline_year_1_arm_1: 0.0372 (95% CI: [0.0322, 0.0421])
# avgPFCthick_QA.baseline_year_1_arm_1: 0.0198 (95% CI: [0.0159, 0.0238])
# rsfmri_c_ngd_cgc_ngd_cgc_QA.baseline_year_1_arm_1: 0.0189 (95% CI: [0.0151, 0.0226])
# rsfmri_c_ngd_dt_ngd_dt_QA.baseline_year_1_arm_1: 0.0260 (95% CI: [0.0214, 0.0305])


# Transformer minority oversampling

# ===== Final Report =====
# Site 21: Accuracy = 0.858, Precision = 0.226, Recall = 0.500, NPV = 0.963, AUC = 0.771, Specificity = 0.882, F1 = 0.311
# Site 11: Accuracy = 0.829, Precision = 0.111, Recall = 1.000, NPV = 1.000, AUC = 1.000, Specificity = 0.825, F1 = 0.200
# Site 4: Accuracy = 0.801, Precision = 0.096, Recall = 0.357, NPV = 0.961, AUC = 0.686, Specificity = 0.824, F1 = 0.152
# Site 5: Accuracy = 0.867, Precision = 0.133, Recall = 0.400, NPV = 0.971, AUC = 0.586, Specificity = 0.887, F1 = 0.200
# Site 6: Accuracy = 0.887, Precision = 0.100, Recall = 0.600, NPV = 0.991, AUC = 0.760, Specificity = 0.893, F1 = 0.171
# Site 20: Accuracy = 0.804, Precision = 0.188, Recall = 0.462, NPV = 0.949, AUC = 0.710, Specificity = 0.833, F1 = 0.267
# Site 1: Accuracy = 0.767, Precision = 0.091, Recall = 0.333, NPV = 0.951, AUC = 0.608, Specificity = 0.794, F1 = 0.143
# Site 18: Accuracy = 0.832, Precision = 0.217, Recall = 0.833, NPV = 0.989, AUC = 0.885, Specificity = 0.832, F1 = 0.345
# Site 3: Accuracy = 0.835, Precision = 0.040, Recall = 0.250, NPV = 0.978, AUC = 0.464, Specificity = 0.850, F1 = 0.069
# Site 12: Accuracy = 0.867, Precision = 0.208, Recall = 0.625, NPV = 0.979, AUC = 0.826, Specificity = 0.879, F1 = 0.312
# Site 14: Accuracy = 0.911, Precision = 0.125, Recall = 0.600, NPV = 0.991, AUC = 0.797, Specificity = 0.917, F1 = 0.207
# Site 10: Accuracy = 0.827, Precision = 0.261, Recall = 0.500, NPV = 0.945, AUC = 0.764, Specificity = 0.860, F1 = 0.343
# Site 16: Accuracy = 0.900, Precision = 0.146, Recall = 0.700, NPV = 0.992, AUC = 0.952, Specificity = 0.904, F1 = 0.241
# Site 13: Accuracy = 0.761, Precision = 0.091, Recall = 1.000, NPV = 1.000, AUC = 0.961, Specificity = 0.755, F1 = 0.167
# Site 9: Accuracy = 0.784, Precision = 0.103, Recall = 0.500, NPV = 0.971, AUC = 0.738, Specificity = 0.797, F1 = 0.171
# Site 2: Accuracy = 0.831, Precision = 0.029, Recall = 0.250, NPV = 0.984, AUC = 0.649, Specificity = 0.842, F1 = 0.051
# Site 15: Accuracy = 0.729, Precision = 0.000, Recall = 0.000, NPV = 0.980, AUC = 0.546, Specificity = 0.740, F1 = 0.000
# Site 19: Accuracy = 0.877, Precision = 0.160, Recall = 0.571, NPV = 0.982, AUC = 0.730, Specificity = 0.888, F1 = 0.250
# Site 8: Accuracy = 0.860, Precision = 0.083, Recall = 0.333, NPV = 0.975, AUC = 0.556, Specificity = 0.878, F1 = 0.133
# Site 17: Accuracy = 0.872, Precision = 0.062, Recall = 1.000, NPV = 1.000, AUC = 0.991, Specificity = 0.871, F1 = 0.118
# Site 7: Accuracy = 0.794, Precision = 0.067, Recall = 1.000, NPV = 1.000, AUC = 1.000, Specificity = 0.791, F1 = 0.125
# Site 22: Accuracy = 0.875, Precision = 0.000, Recall = 0.000, NPV = 1.000, AUC = nan, Specificity = 0.875, F1 = 0.000

# ===== LOSO Summary (Average ± 95% CI) =====
# Accuracy:                0.835 [0.814, 0.856]
# Precision (PPV):         0.115 [0.083, 0.147]
# Recall (Sensitivity):    0.537 [0.405, 0.670]
# NPV:                     0.980 [0.972, 0.987]
# AUC:                     0.761 [0.688, 0.834]
# Specificity:             0.846 [0.825, 0.867]
# F1 Score:                0.181 [0.136, 0.225]

# === Average Cross-Sectional Feature Importances (SHAP) ===
# rel_family_id: 0.0018 (95% CI: [0.0012, 0.0024])
# demo_sex_v2: 0.0639 (95% CI: [0.0545, 0.0733])
# race_ethnicity: 0.0347 (95% CI: [0.0246, 0.0447])
# acs_raked_propensity_score: 0.0190 (95% CI: [0.0151, 0.0229])
# speechdelays: 0.0142 (95% CI: [0.0111, 0.0173])
# motordelays: 0.0172 (95% CI: [0.0127, 0.0216])
# fam_history_8_yes_no: 0.0148 (95% CI: [0.0125, 0.0171])

# === Average Longitudinal Feature Importances (SHAP) ===
# interview_age.baseline_year_1_arm_1: 0.0237 (95% CI: [0.0209, 0.0266])
# KSADSintern.baseline_year_1_arm_1: 0.0349 (95% CI: [0.0300, 0.0398])
# nihtbx_cryst_agecorrected.baseline_year_1_arm_1: 0.0400 (95% CI: [0.0339, 0.0461])
# ACEs.baseline_year_1_arm_1: 0.0468 (95% CI: [0.0425, 0.0511])
# avgPFCthick_QA.baseline_year_1_arm_1: 0.0468 (95% CI: [0.0395, 0.0540])
# rsfmri_c_ngd_cgc_ngd_cgc_QA.baseline_year_1_arm_1: 0.0405 (95% CI: [0.0334, 0.0476])
# rsfmri_c_ngd_dt_ngd_dt_QA.baseline_year_1_arm_1: 0.0419 (95% CI: [0.0355, 0.0483])


#TRANSFORMER data imputation knn + minority over sampling
# ===== Final Report =====
# Site 21: Accuracy = 0.853, Precision = 0.179, Recall = 0.357, NPV = 0.953, AUC = 0.783, Specificity = 0.887, F1 = 0.238
# Site 11: Accuracy = 0.879, Precision = 0.150, Recall = 1.000, NPV = 1.000, AUC = 0.990, Specificity = 0.876, F1 = 0.261
# Site 4: Accuracy = 0.769, Precision = 0.095, Recall = 0.429, NPV = 0.963, AUC = 0.648, Specificity = 0.787, F1 = 0.156
# Site 5: Accuracy = 0.875, Precision = 0.143, Recall = 0.400, NPV = 0.972, AUC = 0.663, Specificity = 0.896, F1 = 0.211
# Site 6: Accuracy = 0.895, Precision = 0.077, Recall = 0.400, NPV = 0.987, AUC = 0.642, Specificity = 0.905, F1 = 0.129
# Site 20: Accuracy = 0.852, Precision = 0.239, Recall = 0.423, NPV = 0.948, AUC = 0.759, Specificity = 0.887, F1 = 0.306
# Site 1: Accuracy = 0.699, Precision = 0.121, Recall = 0.667, NPV = 0.971, AUC = 0.696, Specificity = 0.701, F1 = 0.205
# Site 18: Accuracy = 0.788, Precision = 0.200, Recall = 1.000, NPV = 1.000, AUC = 0.980, Specificity = 0.776, F1 = 0.333
# Site 3: Accuracy = 0.841, Precision = 0.077, Recall = 0.500, NPV = 0.986, AUC = 0.559, Specificity = 0.850, F1 = 0.133
# Site 12: Accuracy = 0.824, Precision = 0.182, Recall = 0.750, NPV = 0.985, AUC = 0.895, Specificity = 0.828, F1 = 0.293
# Site 14: Accuracy = 0.918, Precision = 0.136, Recall = 0.600, NPV = 0.991, AUC = 0.832, Specificity = 0.925, F1 = 0.222
# Site 10: Accuracy = 0.805, Precision = 0.231, Recall = 0.500, NPV = 0.944, AUC = 0.732, Specificity = 0.835, F1 = 0.316
# Site 16: Accuracy = 0.925, Precision = 0.189, Recall = 0.700, NPV = 0.993, AUC = 0.928, Specificity = 0.930, F1 = 0.298
# Site 13: Accuracy = 0.785, Precision = 0.100, Recall = 1.000, NPV = 1.000, AUC = 0.975, Specificity = 0.779, F1 = 0.182
# Site 9: Accuracy = 0.828, Precision = 0.095, Recall = 0.333, NPV = 0.965, AUC = 0.751, Specificity = 0.852, F1 = 0.148
# Site 2: Accuracy = 0.854, Precision = 0.033, Recall = 0.250, NPV = 0.984, AUC = 0.699, Specificity = 0.865, F1 = 0.059
# Site 15: Accuracy = 0.789, Precision = 0.036, Recall = 0.500, NPV = 0.990, AUC = 0.733, Specificity = 0.794, F1 = 0.067
# Site 19: Accuracy = 0.810, Precision = 0.105, Recall = 0.571, NPV = 0.981, AUC = 0.743, Specificity = 0.819, F1 = 0.178
# Site 8: Accuracy = 0.914, Precision = 0.143, Recall = 0.333, NPV = 0.977, AUC = 0.456, Specificity = 0.933, F1 = 0.200
# Site 17: Accuracy = 0.902, Precision = 0.080, Recall = 1.000, NPV = 1.000, AUC = 0.983, Specificity = 0.901, F1 = 0.148
# Site 7: Accuracy = 0.853, Precision = 0.091, Recall = 1.000, NPV = 1.000, AUC = 1.000, Specificity = 0.851, F1 = 0.167
# Site 22: Accuracy = 0.750, Precision = 0.000, Recall = 0.000, NPV = 1.000, AUC = nan, Specificity = 0.750, F1 = 0.000

# ===== LOSO Summary (Average ± 95% CI) =====
# Accuracy:                0.837 [0.811, 0.863]
# Precision (PPV):         0.123 [0.095, 0.151]
# Recall (Sensitivity):    0.578 [0.452, 0.703]
# NPV:                     0.981 [0.973, 0.989]
# AUC:                     0.783 [0.713, 0.853]
# Specificity:             0.847 [0.819, 0.874]
# F1 Score:                0.193 [0.154, 0.232]

# === Average Cross-Sectional Feature Importances (SHAP) ===
# rel_family_id: 0.0018 (95% CI: [0.0011, 0.0025])
# demo_sex_v2: 0.0653 (95% CI: [0.0544, 0.0762])
# race_ethnicity: 0.0347 (95% CI: [0.0237, 0.0457])
# acs_raked_propensity_score: 0.0206 (95% CI: [0.0152, 0.0260])
# speechdelays: 0.0131 (95% CI: [0.0099, 0.0163])
# motordelays: 0.0127 (95% CI: [0.0101, 0.0153])
# fam_history_8_yes_no: 0.0159 (95% CI: [0.0125, 0.0193])

# === Average Longitudinal Feature Importances (SHAP) ===
# interview_age.baseline_year_1_arm_1: 0.0258 (95% CI: [0.0224, 0.0291])
# KSADSintern.baseline_year_1_arm_1: 0.0370 (95% CI: [0.0301, 0.0440])
# nihtbx_cryst_agecorrected.baseline_year_1_arm_1: 0.0362 (95% CI: [0.0313, 0.0410])
# ACEs.baseline_year_1_arm_1: 0.0513 (95% CI: [0.0441, 0.0585])
# avgPFCthick_QA.baseline_year_1_arm_1: 0.0463 (95% CI: [0.0386, 0.0541])
# rsfmri_c_ngd_cgc_ngd_cgc_QA.baseline_year_1_arm_1: 0.0422 (95% CI: [0.0329, 0.0515])
# rsfmri_c_ngd_dt_ngd_dt_QA.baseline_year_1_arm_1: 0.0430 (95% CI: [0.0352, 0.0509])