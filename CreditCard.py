# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For creating visualisations
import seaborn as sns  # For enhanced visualisations
import joblib  # For saving trained models

# Import scikit-learn modules for data preprocessing
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler  # For standardising features

# Import scikit-learn metrics for model evaluation
from sklearn.metrics import (
    classification_report,  # For generating classification reports with precision, recall, f1-score
    confusion_matrix,  # For creating confusion matrices
    roc_curve,  # For calculating Receiver Operating Characteristic curves
    auc,  # For calculating Area Under Curve
    silhouette_score,  # For evaluating clustering performance
    accuracy_score  # For calculating accuracy
)

# Import supervised learning models
from sklearn.linear_model import LogisticRegression  # Linear classifier with logistic function
from sklearn.ensemble import RandomForestClassifier  # Ensemble method using multiple decision trees
from sklearn.tree import DecisionTreeClassifier  # Single decision tree classifier
from sklearn.neighbors import KNeighborsClassifier  # k-nearest neighbours classifier
from sklearn.svm import SVC  # Support Vector Machine classifier
from xgboost import XGBClassifier  # XGBoost gradient boosting classifier

# Import unsupervised learning models
from sklearn.cluster import KMeans, DBSCAN  # Clustering algorithms

#  Load dataset
df = pd.read_csv("creditcard.csv")  # Read the CSV file into a pandas DataFrame
print("Dataset loaded:", df.shape)  # Display the dimensions of the dataset

# Separate features (X) and target variable (y)
X = df.drop("Class", axis=1)  # Features - all columns except 'Class'
y = df["Class"]  # Target variable - the 'Class' column indicating fraud or not

# Preprocess data
# Split the data into training (80%) and testing (20%) sets with fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise the scaler object for standardising features
scaler = StandardScaler()

# Fit the scaler to the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data and transform
X_test_scaled = scaler.transform(X_test)  # Transform testing data using the same scaling
X_full_scaled = scaler.fit_transform(X)  # Scale the full dataset for unsupervised learning

# Save the scaler for future use
joblib.dump(scaler, "creditcard_scaler.pkl")
print("Scaler saved.")  # Confirmation message

# Define and train supervised models
# Create a dictionary of models to train and evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),  # Linear model with increased iterations
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),  # Ensemble of 100 decision trees
    "Decision Tree": DecisionTreeClassifier(random_state=42),  # Single decision tree
    "KNN": KNeighborsClassifier(n_neighbors=5),  # k-NN with 5 neighbours
    "SVM": SVC(probability=True, random_state=42),  # Support Vector Machine with probability estimates
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')  # Gradient boosting
}

# Create a list to store performance metrics for all models
metrics_summary = []

# Loop through each model, train it, and evaluate its performance
for name, model in models.items():
    print(f"/nTraining {name}")  # Display which model is being trained
    model.fit(X_train_scaled, y_train)  # Train the model on the scaled training data
    
    # Save the trained model to a file
    filename = f"{name.lower().replace(' ', '_')}_credit_model.pkl"  # Create filename
    joblib.dump(model, filename)  # Save model
    print(f"Saved model: {filename}")  # Confirmation message
    
    # Make predictions on the test data
    y_pred = model.predict(X_test_scaled)  # Predict class labels
    
    # Get prediction probabilities (handled differently depending on model type)
    if hasattr(model, "predict_proba"):  # If model supports probability predictions
        y_proba = model.predict_proba(X_test_scaled)[:, 1]  # Get probability of positive class
    elif hasattr(model, "decision_function"):  # For models like SVM without direct probability
        y_proba = model.decision_function(X_test_scaled)  # Get decision function values
    else:
        # For models without probability outputs, use predictions directly
        y_proba = y_pred  # Fallback to binary predictions
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    fpr, tpr, _ = roc_curve(y_test, y_proba)  # Calculate ROC curve points
    roc_auc = auc(fpr, tpr)  # Calculate area under ROC curve
    
    # Store metrics for later comparison
    metrics_summary.append({
        "Model": name,
        "Accuracy": accuracy,
        "AUC": roc_auc
    })
    
    # Print classification report with precision, recall, f1-score
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create and save confusion matrix visualisation
    cm = confusion_matrix(y_test, y_pred)  # Calculate confusion matrix
    plt.figure(figsize=(8, 6))  # Set figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Create heatmap
    plt.title(f'{name} - Confusion Matrix')  # Add title
    plt.xlabel("Predicted")  # Label x-axis
    plt.ylabel("Actual")  # Label y-axis
    plt.savefig(f"{name.lower().replace(' ', '_')}_confusion_creditcard.png")  # Save figure
    plt.close()  # Close the figure to free memory
    
    # Create and save ROC curve visualisation
    plt.figure(figsize=(8, 6))  # Set figure size
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')  # Plot ROC curve
    plt.plot([0, 1], [0, 1], linestyle='--')  # Plot diagonal reference line
    plt.title(f'{name} - ROC Curve')  # Add title
    plt.xlabel("False Positive Rate")  # Label x-axis
    plt.ylabel("True Positive Rate")  # Label y-axis
    plt.legend()  # Add legend
    plt.savefig(f"{name.lower().replace(' ', '_')}_roc_creditcard.png")  # Save figure
    plt.close()  # Close the figure to free memory

# Save supervised model scores and create comparison plots
comparison_df = pd.DataFrame(metrics_summary)  # Convert metrics to DataFrame
comparison_df.to_csv("creditcard_model_comparison.csv", index=False)  # Save metrics to CSV

# Create and save accuracy comparison bar chart
plt.figure(figsize=(10, 6))  # Set figure size
sns.barplot(x='Accuracy', y='Model', data=comparison_df)  # Create bar plot
plt.title("Model Accuracy Comparison")  # Add title
plt.xlim(0, 1)  # Set x-axis limits
plt.savefig("creditcard_model_accuracy_comparison.png")  # Save figure
plt.close()  # Close the figure to free memory

# Create and save AUC comparison bar chart
plt.figure(figsize=(10, 6))  # Set figure size
sns.barplot(x='AUC', y='Model', data=comparison_df)  # Create bar plot
plt.title("Model AUC Comparison")  # Add title
plt.xlim(0, 1)  # Set x-axis limits
plt.savefig("creditcard_model_auc_comparison.png")  # Save figure
plt.close()  # Close the figure to free memory

# Unsupervised models
# Create a dictionary to store clustering model results
clustering_summary = {
    "Model": [],  # Name of clustering algorithm
    "Number of Clusters": [],  # How many clusters were found
    "Has Outliers": [],  # Whether the algorithm detected outliers
    "Silhouette Score": []  # Quality metric for clustering
}

# KMeans clustering
print("/nRunning KMeans")  # Display which model is being trained
kmeans = KMeans(n_clusters=2, random_state=42)  # Initialise KMeans with 2 clusters
kmeans_labels = kmeans.fit_predict(X_full_scaled)  # Fit model and get cluster labels
joblib.dump(kmeans, "kmeans_credit_model.pkl")  # Save trained model

# Calculate metrics for KMeans
n_clusters_kmeans = len(set(kmeans_labels))  # Count number of unique clusters
score_kmeans = silhouette_score(X_full_scaled, kmeans_labels)  # Calculate silhouette score

# Create and save KMeans cluster visualisation
plt.figure(figsize=(8, 6))  # Set figure size
plt.scatter(X_full_scaled[:, 0], X_full_scaled[:, 1], c=kmeans_labels, cmap='viridis', s=1)  # Plot clusters
plt.title("KMeans Clusters")  # Add title
plt.savefig("kmeans_clusters_creditcard.png")  # Save figure
plt.close()  # Close the figure to free memory

# Store KMeans results
clustering_summary["Model"].append("KMeans")  # Add model name
clustering_summary["Number of Clusters"].append(n_clusters_kmeans)  # Add number of clusters
clustering_summary["Has Outliers"].append("No")  # KMeans doesn't identify outliers
clustering_summary["Silhouette Score"].append(score_kmeans)  # Add silhouette score

# KMeans++ clustering (improved initialisation method)
print("Running KMeans++")  # Display which model is being trained
kmeans_pp = KMeans(n_clusters=2, init='k-means++', random_state=42)  # Initialise KMeans++ with 2 clusters
kmeans_pp_labels = kmeans_pp.fit_predict(X_full_scaled)  # Fit model and get cluster labels
joblib.dump(kmeans_pp, "kmeans_plus_plus_credit_model.pkl")  # Save trained model

# Calculate metrics for KMeans++
n_clusters_kmeanspp = len(set(kmeans_pp_labels))  # Count number of unique clusters
score_kmeanspp = silhouette_score(X_full_scaled, kmeans_pp_labels)  # Calculate silhouette score

# Create and save KMeans++ cluster visualisation
plt.figure(figsize=(8, 6))  # Set figure size
plt.scatter(X_full_scaled[:, 0], X_full_scaled[:, 1], c=kmeans_pp_labels, cmap='viridis', s=1)  # Plot clusters
plt.title("KMeans++ Clusters")  # Add title
plt.savefig("kmeans_plus_plus_clusters_creditcard.png")  # Save figure
plt.close()  # Close the figure to free memory

# Store KMeans++ results
clustering_summary["Model"].append("KMeans++")  # Add model name
clustering_summary["Number of Clusters"].append(n_clusters_kmeanspp)  # Add number of clusters
clustering_summary["Has Outliers"].append("No")  # KMeans++ doesn't identify outliers
clustering_summary["Silhouette Score"].append(score_kmeanspp)  # Add silhouette score

# DBSCAN clustering (Density-Based Spatial Clustering of Applications with Noise)
print("Running DBSCAN")  # Display which model is being trained
best_score = -1  # Initialise best silhouette score
best_eps = None  # Initialise best epsilon parameter
best_labels = None  # Initialise best cluster labels

# Try different epsilon values to find the best clustering
for eps in np.arange(0.5, 2.0, 0.1):  # Test eps values from 0.5 to 1.9 in steps of 0.1
    db = DBSCAN(eps=eps, min_samples=5).fit(X_full_scaled)  # Create and fit DBSCAN model
    labels = db.labels_  # Get cluster labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Count clusters (excluding noise)
    
    # Only calculate silhouette score if there are multiple clusters and not all points are outliers
    if n_clusters > 1 and -1 in labels and sum(labels != -1) > 0:
        # Only use non-noise points for silhouette calculation
        non_noise_points = X_full_scaled[labels != -1]  # Get points not classified as noise
        non_noise_labels = labels[labels != -1]  # Get labels for non-noise points
        if len(set(non_noise_labels)) > 1:  # Need at least 2 clusters for silhouette score
            try:
                score = silhouette_score(non_noise_points, non_noise_labels)  # Calculate silhouette score
                if score > best_score:  # If this is the best score so far
                    best_score = score  # Update best score
                    best_eps = eps  # Update best epsilon
                    best_labels = labels  # Update best labels
            except:
                continue  # Skip if silhouette calculation fails

# If we found a good DBSCAN configuration
if best_labels is not None:
    final_dbscan = DBSCAN(eps=best_eps, min_samples=5)  # Create final DBSCAN with best parameters
    final_dbscan.fit(X_full_scaled)  # Fit the model
    joblib.dump(final_dbscan, "dbscan_credit_model.pkl")  # Save trained model
    
    # Create and save DBSCAN cluster visualisation
    plt.figure(figsize=(8, 6))  # Set figure size
    plt.scatter(X_full_scaled[:, 0], X_full_scaled[:, 1], c=best_labels, cmap='viridis', s=1)  # Plot clusters
    plt.title(f"DBSCAN Clusters (eps={best_eps:.1f})")  # Add title with epsilon value
    plt.savefig("dbscan_clusters_creditcard.png")  # Save figure
    plt.close()  # Close the figure to free memory
    
    # Calculate DBSCAN metrics
    num_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)  # Count clusters (excluding noise)
    has_outliers = "Yes" if -1 in best_labels else "No"  # Check if DBSCAN detected outliers
    
    # Store DBSCAN results
    clustering_summary["Model"].append("DBSCAN")  # Add model name
    clustering_summary["Number of Clusters"].append(num_clusters)  # Add number of clusters
    clustering_summary["Has Outliers"].append(has_outliers)  # Whether DBSCAN found outliers
    clustering_summary["Silhouette Score"].append(best_score)  # Add silhouette score
else:
    # If no good DBSCAN configuration was found
    clustering_summary["Model"].append("DBSCAN")  # Add model name
    clustering_summary["Number of Clusters"].append("N/A")  # No valid clusters found
    clustering_summary["Has Outliers"].append("N/A")  # Cannot determine
    clustering_summary["Silhouette Score"].append("N/A")  # No valid score

# Save clustering summary and create comparison plots
clustering_df = pd.DataFrame(clustering_summary)  # Convert results to DataFrame
clustering_df.to_csv("creditcard_clustering_comparison.csv", index=False)  # Save results to CSV

# Create and save bar chart comparing number of clusters
plt.figure(figsize=(8, 6))  # Set figure size
# Convert to numeric if necessary
clustering_df["Number of Clusters"] = pd.to_numeric(clustering_df["Number of Clusters"], errors='coerce')  # Convert to numeric
sns.barplot(data=clustering_df[clustering_df["Number of Clusters"].notna()], x='Number of Clusters', y='Model')  # Create bar plot
plt.title("Number of Clusters per Clustering Method")  # Add title
plt.xlabel("Clusters")  # Label x-axis
plt.ylabel("Model")  # Label y-axis
plt.savefig("clustering_num_clusters_comparison.png")  # Save figure
plt.close()  # Close the figure to free memory

# Filter to valid silhouette scores and create comparison chart
valid_silhouette_df = clustering_df[clustering_df["Silhouette Score"] != "N/A"].copy()  # Filter out invalid scores
valid_silhouette_df["Silhouette Score"] = pd.to_numeric(valid_silhouette_df["Silhouette Score"], errors='coerce')  # Convert to numeric

# If we have valid silhouette scores, create comparison chart
if not valid_silhouette_df.empty:
    plt.figure(figsize=(8, 6))  # Set figure size
    sns.barplot(data=valid_silhouette_df, x='Silhouette Score', y='Model')  # Create bar plot
    plt.title("Silhouette Score per Clustering Method")  # Add title
    plt.xlabel("Score")  # Label x-axis
    plt.ylabel("Model")  # Label y-axis
    plt.savefig("clustering_silhouette_score_comparison.png")  # Save figure
    plt.close()  # Close the figure to free memory

print("/nAll models trained, saved, and compared. All plots and CSVs created.")  # Final confirmation message