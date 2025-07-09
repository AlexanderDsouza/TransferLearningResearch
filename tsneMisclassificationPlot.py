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



from sklearn.neighbors import NearestNeighbors
import pandas as pd

def compute_unusualness(file_path, k=5):
    features_scaled, misclass, sids = preprocess_data(file_path)

    # Fit Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(features_scaled)
    distances, _ = nbrs.kneighbors(features_scaled)

    # Skip the first column (distance to itself = 0), take mean of the next k distances
    mean_distances = distances[:, 1:].mean(axis=1)

    # Create a DataFrame mapping SID to unusualness
    unusualness_df = pd.DataFrame({
        'SID': sids,
        'UnusualnessScore': mean_distances
    })

    # Sort by unusualness (descending = most unusual at top)
    unusualness_df = unusualness_df.sort_values(by='UnusualnessScore', ascending=False).reset_index(drop=True)

    return unusualness_df




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

def merge_data_keep_sid(usable_outcomes, cueB_minus_cueA):
    """Merge usable outcomes with CueB - CueA differences and remove identical rows."""
    merged_df_with_SID = usable_outcomes.merge(cueB_minus_cueA, on='SID', how='inner')
    merged_df_with_SID = merged_df_with_SID.drop_duplicates()
    return merged_df_with_SID.drop(columns=['Chg_BPRS'])

def preprocess_data(file_path, misclass_csv_path='sid_misclass_summary2.csv'):
    """
    Load and preprocess the data from Excel and CSV files.
    Returns features_scaled, misclass, sids for downstream use.
    """

    # Load sheets
    outcome_df = pd.read_excel(file_path, sheet_name='outcomes')
    scanner_ids = pd.read_excel(file_path, sheet_name='scannerid')
    fmrifeatures_df = pd.read_excel(file_path, sheet_name='rawfmrifeatures')
    demographics_df = pd.read_excel(file_path, sheet_name='demographics')

    # Split by scanner EP1 or EP2
    subset_scanner_ids_EP1 = scanner_ids[scanner_ids['EP1or2'] == 1]
    subset_scanner_ids_EP2 = scanner_ids[scanner_ids['EP1or2'] == 2]

    usable_outcomes_EP1 = outcome_df[outcome_df['SID'].isin(subset_scanner_ids_EP1['SID'])]
    usable_outcomes_EP2 = outcome_df[outcome_df['SID'].isin(subset_scanner_ids_EP2['SID'])]

    fmrifeatures_df_EP1 = fmrifeatures_df[fmrifeatures_df['SID'].isin(subset_scanner_ids_EP1['SID'])]
    fmrifeatures_df_EP2 = fmrifeatures_df[fmrifeatures_df['SID'].isin(subset_scanner_ids_EP2['SID'])]

    demographics_df_EP1 = demographics_df[demographics_df['SID'].isin(subset_scanner_ids_EP1['SID'])].drop(columns='Dx')
    demographics_df_EP2 = demographics_df[demographics_df['SID'].isin(subset_scanner_ids_EP2['SID'])].drop(columns='Dx')

    # Calculate cue differences
    cueB_minus_cueA_EP1 = calculate_cue_differences(fmrifeatures_df_EP1)
    cueB_minus_cueA_EP2 = calculate_cue_differences(fmrifeatures_df_EP2)

    # Merge cue differences with demographics
    cueB_minus_cueA_EP1 = cueB_minus_cueA_EP1.merge(demographics_df_EP1, on='SID', how='inner')
    cueB_minus_cueA_EP2 = cueB_minus_cueA_EP2.merge(demographics_df_EP2, on='SID', how='inner')
    cueB_minus_cueA_EP2 = cueB_minus_cueA_EP2.drop_duplicates(subset="SID", keep="first")

    # Merge usable outcomes with cue differences
    ep1_data = merge_data(usable_outcomes_EP1, cueB_minus_cueA_EP1)
    ep2_data = merge_data(usable_outcomes_EP2, cueB_minus_cueA_EP2)
    ep2_data_sid = merge_data_keep_sid(usable_outcomes_EP2, cueB_minus_cueA_EP2).drop_duplicates(subset="SID", keep="first")

    # Load misclassification data
    misclass_df = pd.read_csv(misclass_csv_path)

    # Merge EP2 data with misclassification info
    tsne_data = ep2_data_sid.merge(misclass_df, on='SID', how='inner')

    # Extract SID and misclassification percent
    sids = tsne_data['SID']
    misclass = tsne_data['misclass_percent']

    # Drop non-feature columns
    features = tsne_data.drop(columns=['SID', 'misclass_percent', 'avg_pred_prob'])

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, misclass, sids


def tsne_plot(file_path):
    features_scaled, misclass, sids = preprocess_data(file_path)

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_results = tsne.fit_transform(features_scaled)

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=misclass, cmap='coolwarm', s=60, edgecolor='k')
    plt.colorbar(scatter, label='Misclassification %')
    plt.title('t-SNE of EP2 Data Colored by Misclassification %')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True)

    for i, sid in enumerate(sids):
        plt.text(tsne_results[i, 0] + 0.3, tsne_results[i, 1], str(sid), fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.show()


file_path = '/Users/alexd/Documents/Davidson Research/WholeBrainBetas.xlsx'
unusualness_df = compute_unusualness(file_path, k=5)

print(unusualness_df.head(10))  # Top 10 most unusual SIDs
tsne_plot(file_path)


# Path to your file
csv_path = "sid_misclass_summary2.csv"

# If the file exists, read it and merge with unusualness_df
if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path)
    
    # Merge on the 'SID' column (adjust if your key column is named differently)
    merged_df = pd.merge(existing_df, unusualness_df, on="SID", how="outer")
else:
    # If file doesn't exist, just use unusualness_df
    merged_df = unusualness_df

# Save the result back to the CSV (overwrite with updated content)
merged_df.to_csv(csv_path, index=False)




