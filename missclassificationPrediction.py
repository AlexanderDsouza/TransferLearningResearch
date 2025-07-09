
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os


# Load the data
features_df = pd.read_excel("FeaturesforAlex.xlsx")
misclass_df = pd.read_csv("sid_misclass_summary2.csv")




# Merge on 'sid'
merged_df = pd.merge(misclass_df, features_df, on='SID')



# Separate target and features
target = merged_df['misclass_percent']
features = merged_df.drop(columns=['SID', 'misclass_percent','Diagnosis','BL_MedsType'])

# Compute correlations
correlations = features.corrwith(target).sort_values(key=abs, ascending=False)
# Display top correlated features
print("Top features most correlated with misclassification percentage:")
print(correlations)


features_df = pd.read_excel("FeaturesforAlex.xlsx")
misclass_df = pd.read_csv("sid_misclass_summary2.csv")
merged_df = pd.merge(misclass_df, features_df, on='SID')

# Separate target and features
target = merged_df['misclass_percent']
features = merged_df[['PrimaryRace', 'Age', 'IsMale']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the decision tree regressor
model = DecisionTreeRegressor(random_state=42, max_depth=3)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Mean Squared Error: ", mse)
print("R^2 Score: ", r2)


feature_importances = model.feature_importances_

# Create a DataFrame to display the importance of each feature
importance_df = pd.DataFrame({
    'Feature': features.columns,
    'Importance': feature_importances
})

# Sort the features by importance (descending order)
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the top features
print(importance_df)
# Visualize the decision tree
plt.figure(figsize=(15,10))
tree.plot_tree(model, filled=True, feature_names=features.columns, fontsize=10)

desktop_path = os.path.expanduser("~/Desktop/demo_vs_misclassification_rate.png")
plt.savefig(desktop_path)

plt.show()



# Load and merge
features_df = pd.read_excel("FeaturesforAlex.xlsx")
misclass_df = pd.read_csv("sid_misclass_summary2.csv")


# Create scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(misclass_df['UnusualnessScore'], misclass_df['misclass_percent'], alpha=0.7)

# Label each point with SID
for i, row in misclass_df.iterrows():
    plt.text(row['UnusualnessScore'], row['misclass_percent'], str(row['SID']),
             fontsize=8, alpha=0.8)

# Titles and labels
plt.title('Unusualness Score vs Misclassification Percent (Labeled by SID)')
plt.xlabel('Unusualness Score')
plt.ylabel('Misclassification Percent')
plt.grid(True)
plt.tight_layout()

desktop_path = os.path.expanduser("~/Desktop/misclassification_vs_unusualness.png")
plt.savefig(desktop_path)

plt.show()


# Merge on 'SID'
merged_df = pd.merge(misclass_df, features_df, on='SID')

# Define features and target
target = merged_df['UnusualnessScore']
features = merged_df[['PrimaryRace', 'Age', 'IsMale']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model training
model = DecisionTreeRegressor(random_state=42, max_depth=3)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output
print("Mean Squared Error: ", mse)
print("R^2 Score: ", r2)

# Feature importance
importance_df = pd.DataFrame({
    'Feature': features.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(importance_df)

# Tree visualization
plt.figure(figsize=(15,10))
tree.plot_tree(model, filled=True, feature_names=features.columns, fontsize=10)
plt.show()
