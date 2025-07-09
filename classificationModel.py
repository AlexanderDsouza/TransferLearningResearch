import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
file_path = '/Users/alexd/Documents/Davidson Research/DataforAlex.xlsx'
data = pd.read_excel(file_path)
data = data.drop(['The NDAR Global Unique Identifier (GUID) for research subject'], axis=1)



# Fill missing values with the mean (for numeric columns)
data_filled = data.fillna(data.mean())
# Remove missing entries
data_cleaned = data.dropna()

X = data_cleaned.drop('Converted', axis=1)
y = data_cleaned['Converted']

# Check class distribution
print("Class distribution:")
print(y.value_counts())

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# Apply SMOTE to the training set
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Initialize the Random Forest model with class weights
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 1, 1: 5}, max_depth=10)

# Create a pipeline that includes scaling and the model
pipeline = make_pipeline(StandardScaler(), model)

# Fit the model using the resampled data
pipeline.fit(X_resampled, y_resampled)

# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, X_resampled, y_resampled, cv=5)  # Using 5-fold cross-validation

print(f"Cross-validation scores: {cross_val_scores}")
print(f"Mean cross-validation score: {cross_val_scores.mean():.2f}")

# Make predictions on the training and test sets
y_train_pred = pipeline.predict(X_resampled)  # Predictions on resampled training set
y_test_pred = pipeline.predict(X_test)

# Calculate accuracy for both sets
train_accuracy = accuracy_score(y_resampled, y_train_pred)  # Use y_resampled for accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Optional: Print classification report and confusion matrix for the test set
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, zero_division=0))

print(f"Number of samples in the test set: {X_test.shape[0]}")
print(f"Number of samples in the training set: {X_train.shape[0]}")

conf_matrix = confusion_matrix(y_test, y_test_pred)


# Optional: Print classification report and confusion matrix for the test set
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, zero_division=0))

print(f"Number of samples in the test set: {X_test.shape[0]}")
print(f"Number of samples in the training set: {X_train.shape[0]}")

conf_matrix = confusion_matrix(y_test, y_test_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Converted (0)', 'Converted (1)'], yticklabels=['Not Converted (0)', 'Converted (1)'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Print classification report for more metrics
print("Classification Report:")
print(classification_report(y_test, y_test_pred, zero_division=0))

# Calculate True Positives, True Negatives, False Positives, and False Negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

# Accuracy for converters (True Positives)
accuracy_converters = tp / (tp + fn) if (tp + fn) > 0 else 0

# Accuracy for non-converters (True Negatives)
accuracy_non_converters = tn / (tn + fp) if (tn + fp) > 0 else 0

# Balanced Accuracy
sensitivity = accuracy_converters  # This is the same as the recall for converters
specificity = accuracy_non_converters  # This is the same as the recall for non-converters
balanced_accuracy = (sensitivity + specificity) / 2

# Calculate ROC AUC Value
roc_auc_value = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

# Print results
print(f"Accuracy for Converters: {accuracy_converters * 100:.2f}%")
print(f"Accuracy for Non-Converters: {accuracy_non_converters * 100:.2f}%")
print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
print(f"ROC AUC Value: {roc_auc_value:.2f}")




# Create and show the correlation matrix (if needed)
# correlation_matrix = data.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
# plt.title('Correlation Matrix')
# plt.show()
