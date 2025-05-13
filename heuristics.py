# Import libraries needed for the project
import pandas as pd  # This lets us work with data in tables
import numpy as np  # numerical calculations
import matplotlib.pyplot as plt  # create plots and graphs 
import seaborn as sns  # This is an extension of matplotlib that makes our plots look nicer and easier to create
import os  # This lets us interact with the operating system, like clearing the terminal screen
import joblib

# Import tools from scikit-learn for splitting data, scaling, and evaluating models
# train_test_split splits data into training and testing sets, cross_val_score checks model performance,
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV  
# GridSearchCV Grid Search helps find the best hyperparameters for a model 
from sklearn.preprocessing import StandardScaler  # This scales numerical data so all features are on the same scale (important for many machine learning models)
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, silhouette_score  
# These are for evaluating models: confusion_matrix shows prediction errors
#classification_report gives precision/recall
# roc_curve and auc measure model performance
# silhouette_score is for clustering

# Import machine learning models for classification (predicting fraud)
from sklearn.linear_model import LogisticRegression  # This is a simple model that predicts probabilities (good for binary classification like fraud/not fraud)
from sklearn.ensemble import RandomForestClassifier  # This is a model that combines many decision trees to make better predictions
from sklearn.tree import DecisionTreeClassifier  # This is a single decision tree model that splits data based on feature values
from sklearn.neighbors import KNeighborsClassifier  # This model predicts based on the nearest data points (using distance)
from sklearn.svm import SVC  # This is a Support Vector Machine, a model that finds the best boundary between classes
from xgboost import XGBClassifier  # This is an XGBoost model, a gradient boosting algorithm that combines weak learners for strong predictions

# Import clustering models for unsupervised learning (finding patterns without labels)
from sklearn.cluster import KMeans, DBSCAN  # KMeans groups data into clusters, DBSCAN finds clusters based on density

# Import a statistical test for checking relationships between categorical data
from scipy.stats import chi2_contingency  # This tests if two categorical variables (like Education and Fraud) are related

# Define a class to manage our fraud analysis system
class FraudAnalysisSystem:
    # This is the constructor method that runs when we create a new FraudAnalysisSystem object
    def __init__(self):
        self.df_original = None  # This will store the original dataset before cleaning
        self.df_cleaned = None  # This will store the dataset after cleaning
        self.X_train = None  # This will store the training features for supervised learning
        self.X_test = None  # This will store the testing features for supervised learning
        self.y_train = None  # This will store the training labels (Fraud or not) for supervised learning
        self.y_test = None  # This will store the testing labels (Fraud or not) for supervised learning
        self.scaler = StandardScaler()  # This creates a StandardScaler object to scale numerical data
        self.unsupervised_data = None  # This will store the data prepared for unsupervised learning
        self.unsupervised_features = None  # This will store the scaled features for unsupervised learning
        self.models = {}  # This dictionary will store trained supervised models (like Random Forest)
        self.model_metrics = {}  # This dictionary will store performance metrics for supervised models
        self.clustering_results = {}  # This dictionary will store results from clustering (like K-Means)

    # This method cleans the dataset to make it ready for analysis
    def clean_data(self, df):
        df_clean = df.copy()  # Make a copy of the input dataset so we don’t change the original
        # Create a dictionary to standardise Education values (e.g., 'U' and 'Uni' both become 'University')
        education_mapping = {
            'U': 'University', 'S': 'School', 'C': 'College',
            'UG': 'University', 'PG': 'University',
            'HS': 'School', 'Uni': 'University',
            'School': 'School', 'College': 'College',
            'University': 'University', 'college': 'College',
            'school': 'School', 'university': 'University',
            'CollegeU': 'College', 'Coll': 'College',
            'UniversityS': 'University'
        }
        df_clean['Education'] = df_clean['Education'].map(education_mapping)  # Apply the mapping to the Education column to standardise values
        df_clean['Education'] = df_clean['Education'].fillna(df_clean['Education'].mode()[0])  # Fill any missing Education values with the most common value (mode)
        # Create a dictionary to standardise Area values (e.g., 'U' becomes 'Urban')
        area_mapping = {'U': 'Urban', 'R': 'Rural', 'u': 'Urban', 'r': 'Rural',
                        'Urban': 'Urban', 'Rural': 'Rural', 'rurak': 'Rural'}
        df_clean['Area'] = df_clean['Area'].map(area_mapping)  # Apply the mapping to the Area column to standardise values
        df_clean['Area'] = df_clean['Area'].fillna(df_clean['Area'].mode()[0])  # Fill any missing Area values with the most common value (mode)
        df_clean['Gender'] = df_clean['Gender'].astype(str).str.lower()  # Convert Gender to lowercase strings to standardise (e.g., 'MALE' becomes 'male')
        # Replace various Gender values with standard ones (e.g., 'm' becomes 'Male')
        df_clean['Gender'] = df_clean['Gender'].replace({
            'm': 'Male', 'f': 'Female',
            'male': 'Male', 'female': 'Female',
            'pns': 'Prefer not to say', 'prefer not to say': 'Prefer not to say',
            'nan': 'Prefer not to say', 'none': 'Prefer not to say',
            '': 'Prefer not to say'
        })
        valid_genders = ['Male', 'Female', 'Prefer not to say']  # Define the valid Gender categories we’ll accept
        # If a Gender value isn’t in the valid list, set it to 'Prefer not to say'
        df_clean.loc[~df_clean['Gender'].isin(valid_genders), 'Gender'] = 'Prefer not to say'
        df_clean['Gender'] = df_clean['Gender'].fillna('Prefer not to say')  # Fill any remaining missing Gender values with 'Prefer not to say'
        # Create a dictionary to standardise Yes/No values (e.g., 'Y' becomes 'Yes', '1' becomes 'Yes')
        yes_no_mapping = {'Y': 'Yes', 'N': 'No', '1': 'Yes', '0': 'No', 'Yes': 'Yes', 'No': 'No'}
        # Loop through Home Owner and Employed columns to apply the Yes/No mapping
        for col in ['Home Owner', 'Employed']:
            df_clean[col] = df_clean[col].map(yes_no_mapping)  # Apply the mapping to the column to standardise values
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])  # Fill any missing values with the most common value (mode)
        # Print unique values in Fraud before cleaning
        print("Unique values in 'Fraud' before cleaning:", df_clean['Fraud'].unique())
        df_clean['Fraud'] = df_clean['Fraud'].astype(str)  # Convert the Fraud column to strings so it matches the mapping keys
        # Map Fraud values to Yes/No, then to 1/0 (e.g., '1' becomes 'Yes', then 'Yes' becomes 1)
        df_clean['Fraud'] = df_clean['Fraud'].map(yes_no_mapping).map({'Yes': 1, 'No': 0})
        # Check if all Fraud values are NaN after mapping (this can happen if the mapping fails)
        if df_clean['Fraud'].isna().all():
            print("Warning: All values in 'Fraud' column are NaN after mapping. Setting default mode to 0.")  # Print a warning if mapping fails
            fraud_mode = 0  # If all values are NaN, set the default fill value to 0 (not fraud)
        else:
            fraud_mode = df_clean['Fraud'].mode().iloc[0]  # Otherwise, use the most common Fraud value (mode)
        # Print unique values in Fraud after mapping
        print("Unique values in 'Fraud' after mapping:", df_clean['Fraud'].unique())
        df_clean['Balance'] = pd.to_numeric(df_clean['Balance'], errors='coerce')  # Convert Balance to numbers, turn non-numbers into NaN
        df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')  # Convert Age to numbers, turn non-numbers into NaN
        df_clean['Income'] = df_clean['Income'].fillna(df_clean['Income'].median())  # Fill missing Income values with the median (middle value)
        df_clean['Balance'] = df_clean['Balance'].fillna(df_clean['Balance'].median())  # Fill missing Balance values with the median
        df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].mean())  # Fill missing Age values with the average (mean)
        df_clean['Colour'] = df_clean['Colour'].fillna(df_clean['Colour'].mode()[0])  # Fill missing Colour values with the most common value (mode)
        df_clean['Fraud'] = df_clean['Fraud'].fillna(fraud_mode)  # Fill any remaining missing Fraud values with the mode we calculated earlier
        # Print unique values in Fraud after filling NaNs
        print("Unique values in 'Fraud' after filling NaNs:", df_clean['Fraud'].unique())
        # Check if there’s an extra column called 'Unnamed: 10' (sometimes happens with CSV files) and remove it
        if 'Unnamed: 10' in df_clean.columns:
            df_clean = df_clean.drop('Unnamed: 10', axis=1)  # Drop the extra column (axis=1 means drop a column, not a row)
        df_clean.to_csv('cleaned_data.csv', index=False)  # Save the cleaned dataset to a new CSV file without the row numbers
        print("✓ Cleaned data saved to 'cleaned_data.csv'")  # Print a confirmation message that the data has been saved
        return df_clean  # Return the cleaned dataset for further use

    # This method creates plots to compare the data before and after cleaning
    def plot_comparisons(self, df_before, df_after):
        # Define all columns to plot
        columns = ['Income', 'Area', 'Employed', 'Education', 'Gender', 'Balance', 'Age', 'Fraud', 'Colour', 'Home Owner']
        
        # Columns that need separate plots for the "Before" state due to clutter
        separate_plot_cols = ['Income', 'Balance', 'Age']
        
        # Split into two groups of 5 columns each
        columns_part1 = columns[:5]  # First 5 columns
        columns_part2 = columns[5:]  # Last 5 columns

        # Get the unique values from the raw data to ensure the "After" plot includes all categories (e.g., NaN)
        unique_values = {}
        for col in separate_plot_cols:
            # Convert raw data to numeric and then to strings with consistent formatting
            raw_data = pd.to_numeric(df_before[col], errors='coerce')
            if col in ['Income', 'Balance']:
                # Format Income and Balance to 1 decimal place to match cleaned data
                unique_vals = raw_data.apply(lambda x: f"{x:.1f}" if not pd.isna(x) else 'nan').astype(str).unique()
                # Sort unique values numerically, keeping 'nan' at the end
                numeric_vals = [float(x) for x in unique_vals if x != 'nan']
                numeric_vals.sort()
                unique_vals = [f"{x:.1f}" for x in numeric_vals] + ['nan']
            else:
                unique_vals = raw_data.astype(str).unique()
                unique_vals = sorted(unique_vals)
            unique_values[col] = unique_vals

        # Create separate plots for Income, Balance, and Age "Before" state
        for col in separate_plot_cols:
            plt.figure(figsize=(15, 6))  # Create a larger figure for better readability
            raw_data = pd.to_numeric(df_before[col], errors='coerce')
            if col in ['Income', 'Balance']:
                raw_data = raw_data.apply(lambda x: f"{x:.1f}" if not pd.isna(x) else 'nan').astype(str)
            else:
                raw_data = raw_data.astype(str)
            sns.countplot(x=raw_data, order=sorted(raw_data.unique()))  # Plot a count of all values with sorted order
            plt.title(f'Before: {col} (Raw Data)')  # Add a title to the plot
            plt.xlabel(col)  # Label the x-axis
            plt.ylabel('Count')  # Label the y-axis
            plt.xticks(rotation=90)  # Rotate x-axis labels to avoid overlap
            plt.tight_layout()  # Adjust the layout
            plt.savefig(f'before_{col.lower()}_raw.png')  # Save the plot as a PNG file
            print(f"Saved separate 'Before' plot for {col} to 'before_{col.lower()}_raw.png'")
            plt.show()  # Display the plot on the screen

        # Create separate plots for Income, Balance, and Age "After" state
        for col in separate_plot_cols:
            plt.figure(figsize=(15, 6))  # Create a larger figure for better readability
            # Ensure the cleaned data is numeric, then convert to strings
            cleaned_data = pd.to_numeric(df_after[col], errors='coerce')
            # Debug: Print unique values to diagnose mismatches
            print(f"\nDebug - {col} unique values in raw data: {unique_values[col]}")
            # Special handling for each column
            if col == 'Age':
                # Round Age to one decimal place (instead of int) to match '17.0' format from raw data
                cleaned_data = cleaned_data.apply(lambda x: f"{x:.1f}" if not pd.isna(x) else 'nan').astype(str)
            elif col in ['Income', 'Balance']:
                # Ensure consistent formatting for Income and Balance
                cleaned_data = cleaned_data.apply(lambda x: f"{x:.1f}" if not pd.isna(x) else 'nan').astype(str)
            else:
                cleaned_data = cleaned_data.astype(str)
            # Debug: Print unique values after formatting
            print(f"Debug - {col} unique values in cleaned data after formatting: {cleaned_data.unique()}")
            # Create a Series with all possible categories (from raw data) to ensure empty categories have count 0
            all_categories = pd.Series(cleaned_data, dtype='category')
            all_categories = all_categories.cat.set_categories(unique_values[col])
            # Plot the count of all values, ensuring categories with count 0 are shown
            sns.countplot(x=all_categories, order=unique_values[col])  # Use order to explicitly set the categories
            plt.title(f'After: {col} (Cleaned Data)')  # Add a title to the plot
            plt.xlabel(col)  # Label the x-axis
            plt.ylabel('Count')  # Label the y-axis
            plt.xticks(rotation=90)  # Rotate x-axis labels to avoid overlap
            plt.tight_layout()  # Adjust the layout
            plt.savefig(f'after_{col.lower()}_cleaned.png')  # Save the plot as a PNG file
            print(f"Saved separate 'After' plot for {col} to 'after_{col.lower()}_cleaned.png'")
            plt.show()  # Display the plot on the screen

        # Plot the first 5 columns (Part 1)
        fig1 = plt.figure(figsize=(25, 10))  # Create a grid of 2 rows and 5 columns for plots, with a size of 25x10 inches
        axes = fig1.subplots(2, 5)
        fig1.suptitle('Data Before and After Cleaning (Part 1)', fontsize=16, y=1.05)  # Add a title to the entire figure
        # Loop through each column to create a plot for it
        for idx, col in enumerate(columns_part1):
            # For the "Before" plot
            if col in separate_plot_cols:
                # Add a placeholder text indicating the separate plot
                axes[0, idx].text(0.5, 0.5, f'See separate plot\n(before_{col.lower()}_raw.png)', 
                                  ha='center', va='center', fontsize=12)
                axes[0, idx].set_title(f'Before: {col}')
                axes[0, idx].axis('off')  # Hide the axes for this placeholder
            else:
                sns.countplot(x=df_before[col].astype(str), ax=axes[0, idx])  # Plot a count of all values
                axes[0, idx].set_title(f'Before: {col}')  # Add a title to the top plot
                axes[0, idx].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
            # For the "After" plot
            if col in separate_plot_cols:
                # Add a placeholder text indicating the separate plot
                axes[1, idx].text(0.5, 0.5, f'See separate plot\n(after_{col.lower()}_cleaned.png)', 
                                  ha='center', va='center', fontsize=12)
                axes[1, idx].set_title(f'After: {col}')
                axes[1, idx].axis('off')  # Hide the axes for this placeholder
            else:
                if df_after[col].dtype in ['int64', 'float64']:
                    sns.histplot(df_after[col].dropna(), ax=axes[1, idx], bins=30)  # Plot a histogram
                    if col == 'Balance':
                        data_min = df_after[col].dropna().min()
                        if data_min > 0:
                            axes[1, idx].set_xscale('log')
                        else:
                            print(f"Warning: Skipping log scale for 'After: {col}' due to zero or negative values (min: {data_min})")
                else:
                    sns.countplot(data=df_after, x=col, ax=axes[1, idx])  # Plot a count of each category
                axes[1, idx].set_title(f'After: {col}')  # Add a title to the bottom plot
                axes[1, idx].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
        plt.tight_layout()  # Adjust the layout
        plt.savefig('data_cleaning_comparison_part1.png')  # Save the comparison plot
        print("Saved Part 1 plot to 'data_cleaning_comparison_part1.png'")
        plt.show()  # Display the plot

        # Plot the last 5 columns (Part 2)
        fig2 = plt.figure(figsize=(25, 10))  # Create a grid of 2 rows and 5 columns for plots
        axes = fig2.subplots(2, 5)
        fig2.suptitle('Data Before and After Cleaning (Part 2)', fontsize=16, y=1.05)  # Add a title to the entire figure
        # Loop through each column to create a plot for it
        for idx, col in enumerate(columns_part2):
            # For the "Before" plot
            if col in separate_plot_cols:
                # Add a placeholder text indicating the separate plot
                axes[0, idx].text(0.5, 0.5, f'See separate plot\n(before_{col.lower()}_raw.png)', 
                                  ha='center', va='center', fontsize=12)
                axes[0, idx].set_title(f'Before: {col}')
                axes[0, idx].axis('off')  # Hide the axes for this placeholder
            else:
                sns.countplot(x=df_before[col].astype(str), ax=axes[0, idx])  # Plot a count of all values
                axes[0, idx].set_title(f'Before: {col}')  # Add a title to the top plot
                axes[0, idx].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
            # For the "After" plot
            if col in separate_plot_cols:
                # Add a placeholder text indicating the separate plot
                axes[1, idx].text(0.5, 0.5, f'See separate plot\n(after_{col.lower()}_cleaned.png)', 
                                  ha='center', va='center', fontsize=12)
                axes[1, idx].set_title(f'After: {col}')
                axes[1, idx].axis('off')  # Hide the axes for this placeholder
            else:
                if df_after[col].dtype in ['int64', 'float64']:
                    sns.histplot(df_after[col].dropna(), ax=axes[1, idx], bins=30)  # Plot a histogram
                    if col == 'Balance':
                        data_min = df_after[col].dropna().min()
                        if data_min > 0:
                            axes[1, idx].set_xscale('log')
                        else:
                            print(f"Warning: Skipping log scale for 'After: {col}' due to zero or negative values (min: {data_min})")
                else:
                    sns.countplot(data=df_after, x=col, ax=axes[1, idx])  # Plot a count of each category
                axes[1, idx].set_title(f'After: {col}')  # Add a title to the bottom plot
                axes[1, idx].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
        plt.tight_layout()  # Adjust the layout
        plt.savefig('data_cleaning_comparison_part2.png')  # Save the comparison plot
        print("Saved Part 2 plot to 'data_cleaning_comparison_part2.png'")
        plt.show()  # Display the plot

    # This method analyses how different features in the dataset relate to fraud
    def analyse_fraud_relationships(self, df):
        df_plot = df.copy()  # Make a copy of the dataset so we don’t change the original while plotting
        df_plot['Fraud'] = df_plot['Fraud'].map({0: 'Not Fraud', 1: 'Fraud'})  # Change Fraud values from 0/1 to 'Not Fraud'/'Fraud' for better labels in plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # Create a 2x2 grid of plots, with a size of 15x10 inches
        sns.boxplot(data=df_plot, x='Fraud', y='Income', ax=axes[0, 0])  # Plot a boxplot of Income vs Fraud status in the top-left position
        axes[0, 0].set_title('Income Distribution by Fraud Status')  # Add a title to the Income boxplot
        sns.boxplot(data=df_plot, x='Fraud', y='Age', ax=axes[0, 1])  # Plot a boxplot of Age vs Fraud status in the top-right position
        axes[0, 1].set_title('Age Distribution by Fraud Status')  # Add a title to the Age boxplot
        sns.boxplot(data=df_plot, x='Fraud', y='Balance', ax=axes[1, 0])  # Plot a boxplot of Balance vs Fraud status in the bottom-left position
        axes[1, 0].set_yscale('log')  # Use a logarithmic scale for Balance because the values might vary a lot
        axes[1, 0].set_title('Balance Distribution by Fraud Status (Log Scale)')  # Add a title to the Balance boxplot
        sns.countplot(data=df_plot, x='Education', hue='Fraud', ax=axes[1, 1])  # Plot a count of Education categories, split by Fraud status, in the bottom-right position
        axes[1, 1].set_title('Education Level by Fraud Status')  # Add a title to the Education count plot
        axes[1, 1].tick_params(axis='x', rotation=45)  # Rotate the x-axis labels by 45 degrees to make them readable
        axes[1, 1].legend(title='Fraud Status')  # Add a legend to show what the colours in the plot represent
        # Add a grid to each plot to make them easier to read
        for ax in axes.flat:
            ax.grid(True, linestyle='--', alpha=0.7)  # Add a dashed grid with some transparency to each subplot
        plt.tight_layout()  # Adjust the layout so the plots don’t overlap
        plt.savefig('fraud_relationships.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen
        print("\nChi-Square Tests for Categorical Features vs Fraud:")  # Print a header for the statistical tests
        categorical_cols = ['Education', 'Gender', 'Area', 'Home Owner', 'Employed']  # List the categorical columns we want to test against Fraud
        # Loop through each categorical column to test its relationship with Fraud
        for col in categorical_cols:
            contingency_table = pd.crosstab(df_plot[col], df_plot['Fraud'])  # Create a table counting how many times each category appears with Fraud/Not Fraud
            chi2, p, _, _ = chi2_contingency(contingency_table)  # Run a chi-square test to see if the variables are statistically related
            print(f"{col} vs Fraud: Chi2 = {chi2:.2f}, p-value = {p:.4f}")  # Print the chi-square statistic and p-value (p < 0.05 means they’re related)

    # This method creates a correlation matrix to show relationships between numerical features
    def plot_correlation_matrix(self, df):
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns  # Get the names of numerical columns (like Income, Age, Balance)
        correlation_matrix = df[numerical_cols].corr()  # Calculate the correlation between all numerical columns (values range from -1 to 1)
        plt.figure(figsize=(10, 8))  # Create a new figure with a size of 10x8 inches
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)  # Plot a heatmap of the correlations, showing the numbers, using a red-blue colour scheme
        plt.title('Correlation Matrix of Numerical Features')  # Add a title to the plot
        plt.savefig('correlation_matrix.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen

    # This method prepares the data for machine learning (either supervised or unsupervised)
    def prepare_data(self, for_supervised=True):
        # If we’re preparing for supervised learning (predicting Fraud)
        if for_supervised:
            X = pd.get_dummies(self.df_cleaned.drop('Fraud', axis=1))  # Convert categorical columns to numerical (e.g., 'Male' becomes 0/1), and drop the Fraud column
            y = self.df_cleaned['Fraud']  # Store the Fraud column as the target variable we want to predict
            # Split the data into training (80%) and testing (20%) sets, using a fixed random seed for reproducibility
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.X_train = self.scaler.fit_transform(self.X_train)  # Scale the training features to have mean 0 and variance 1 (important for many models)
            self.X_test = self.scaler.transform(self.X_test)  # Scale the testing features using the same scaler (don’t fit again, just transform)
        # If we’re preparing for unsupervised learning (clustering)
        else:
            self.unsupervised_data = pd.get_dummies(self.df_cleaned.drop('Fraud', axis=1))  # Convert categorical columns to numerical, drop Fraud since we’re not predicting it
            self.unsupervised_features = StandardScaler().fit_transform(self.unsupervised_data)  # Scale the features for clustering to have mean 0 and variance 1

    # This method plots a confusion matrix to show prediction errors for a model
    def plot_confusion_matrix(self, y_true, y_pred, title):
        plt.figure(figsize=(8, 6))  # Create a new figure with a size of 8x6 inches
        cm = confusion_matrix(y_true, y_pred)  # Calculate the confusion matrix (shows true vs predicted labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Plot the matrix as a heatmap, with numbers shown, using a blue colour scheme
        plt.title(title)  # Add the title to the plot (e.g., 'Random Forest Confusion Matrix')
        plt.ylabel('True Label')  # Label the y-axis as the true labels (Fraud/Not Fraud)
        plt.xlabel('Predicted Label')  # Label the x-axis as the predicted labels
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")  # Save the plot as a PNG file (e.g., 'Random Forest Confusion Matrix' becomes 'random_forest_confusion_matrix.png')
        plt.show()  # Display the plot on the screen

    # This method plots an ROC curve to evaluate a model’s performance
    def plot_roc_curve(self, model, title):
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]  # Get the predicted probabilities for the positive class (Fraud)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)  # Calculate the false positive rate and true positive rate for the ROC curve
        roc_auc = auc(fpr, tpr)  # Calculate the area under the ROC curve (AUC), a measure of model performance
        plt.figure(figsize=(8, 6))  # Create a new figure with a size of 8x6 inches
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')  # Plot the ROC curve with the AUC in the label
        plt.plot([0, 1], [0, 1], 'k--')  # Plot a dashed line from (0,0) to (1,1) as a reference (random guessing)
        plt.xlabel('False Positive Rate')  # Label the x-axis
        plt.ylabel('True Positive Rate')  # Label the y-axis
        plt.title(f'{title} - ROC Curve')  # Add a title to the plot (e.g., 'Random Forest - ROC Curve')
        plt.legend()  # Show the legend with the AUC value
        plt.savefig(f"{title.replace(' ', '_').lower()}_roc_curve.png")  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen

    # This method prints a summary of the dataset with statistics and plots
    def print_data_summary(self):
        print("\nDataset Summary:")  # Print a header for the summary
        print("-" * 50)  # Print a line of dashes to separate sections
        print("\nBasic Information:")  # Print a subheader for basic info
        print(self.df_cleaned.info())  # Print basic info about the dataset (like column names, data types, and missing values)
        print("\nDescriptive Statistics:")  # Print a subheader for statistics
        print(self.df_cleaned.describe())  # Print stats like mean, min, max for numerical columns
        print("\nFraud Distribution:")  # Print a subheader for the Fraud distribution
        value_counts = self.df_cleaned['Fraud'].value_counts(normalize=True) * 100  # Calculate the percentage of Fraud and Not Fraud
        # Loop through the counts to print them in a readable format
        for val, count in value_counts.items():
            status = "Not Fraud" if val == 0 else "Fraud"  # Convert 0/1 to Not Fraud/Fraud for readability
            print(f"{status}: {count:.2f}%")  # Print the percentage for each class
        print("\nClass Imbalance Ratio:")  # Print a subheader for the imbalance ratio
        fraud_ratio = len(self.df_cleaned[self.df_cleaned['Fraud'] == 1]) / len(self.df_cleaned[self.df_cleaned['Fraud'] == 0])  # Calculate the ratio of Fraud to Not Fraud
        print(f"Fraud to Non-Fraud ratio: {fraud_ratio:.3f}")  # Print the ratio to 3 decimal places
        numerical_cols = ['Income', 'Balance', 'Age']  # List the numerical columns we want to plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a 1x3 grid of plots, with a size of 15x5 inches
        # Loop through each numerical column to plot its distribution
        for idx, col in enumerate(numerical_cols):
            sns.histplot(self.df_cleaned[col], ax=axes[idx])  # Plot a histogram of the column to show its distribution
            axes[idx].set_title(f'Distribution of {col}')  # Add a title to the plot (e.g., 'Distribution of Income')
        plt.tight_layout()  # Adjust the layout so the plots don’t overlap
        plt.savefig('numerical_distributions.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen

    # This method trains a Logistic Regression model to predict fraud
    def train_logistic_regression(self):
        print("\n=== Training Logistic Regression ===")  # Print a header to show we’re starting Logistic Regression
        model = LogisticRegression(random_state=42)  # Create a Logistic Regression model with a fixed random seed for reproducibility
        model.fit(self.X_train, self.y_train)  # Train the model on the training data (features and labels)
        y_pred = model.predict(self.X_test)  # Make predictions on the test data
        print("\nClassification Report:")  # Print a subheader for the classification report
        print(classification_report(self.y_test, y_pred))  # Print a detailed report with precision, recall, and F1-score for each class
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)  # Perform 5-fold cross-validation on the training data to check consistency
        print("\nCross-validation scores:", cv_scores)  # Print the scores for each fold
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")  # Print the average score and variability (2 standard deviations)
        self.plot_confusion_matrix(self.y_test, y_pred, "Logistic Regression Confusion Matrix")  # Plot the confusion matrix to show prediction errors
        self.plot_roc_curve(model, "Logistic Regression")  # Plot the ROC curve to evaluate the model’s performance
        self.models['Logistic Regression'] = model  # Store the trained model in the models dictionary for later use
        joblib.dump(model, 'Logistic_regression_model.pkl') 
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]  # Get the predicted probabilities for the Fraud class
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)  # Calculate the false positive and true positive rates for the ROC curve
        self.model_metrics['Logistic Regression'] = {  # Store the model’s performance metrics in the model_metrics dictionary
            'Accuracy': model.score(self.X_test, self.y_test),  # Calculate the accuracy on the test set
            'ROC AUC': auc(fpr, tpr)  # Calculate the area under the ROC curve
        }

    # This method trains a Random Forest model to predict fraud
    def train_random_forest(self):
        print("\n=== Training Random Forest ===")  # Print a header to show we’re starting Random Forest
        param_grid = {  # Define the hyperparameters we want to test for tuning the model
            'n_estimators': [50, 100, 200],  # Number of trees in the forest 
            'max_depth': [None, 10, 20]  # Maximum depth of each tree 
        }
        base_model = RandomForestClassifier(random_state=42)  # Create a Random Forest model with a fixed random seed
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)  # Set up a grid search to find the best hyperparameters, using 5-fold cross-validation
        grid_search.fit(self.X_train, self.y_train)  # Run the grid search on the training data
        model = grid_search.best_estimator_  # Get the best model from the grid search
        print(f"Best parameters: {grid_search.best_params_}")  # Print the best hyperparameters found
        y_pred = model.predict(self.X_test)  # Make predictions on the test data
        print("\nClassification Report:")  # Print a subheader for the classification report
        print(classification_report(self.y_test, y_pred))  # Print a detailed report with precision, recall, and F1-score for each class
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)  # Perform 5-fold cross-validation with the best model
        print("\nCross-validation scores:", cv_scores)  # Print the scores for each fold
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")  # Print the average score and variability
        self.plot_confusion_matrix(self.y_test, y_pred, "Random Forest Confusion Matrix")  # Plot the confusion matrix to show prediction errors
        self.plot_roc_curve(model, "Random Forest")  # Plot the ROC curve to evaluate the model’s performance
        feature_names = pd.get_dummies(self.df_cleaned.drop('Fraud', axis=1)).columns  # Get the names of the features after converting categorical variables
        # Create a DataFrame with the feature names and their importance scores
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)  # Sort by importance (highest to lowest)
        plt.figure(figsize=(10, 6))  # Create a new figure with a size of 10x6 inches
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')  # Plot a bar chart of the top 10 most important features
        plt.title('Top 10 Most Important Features')  # Add a title to the plot
        plt.xlabel('Feature Importance')  # Label the x-axis
        plt.tight_layout()  # Adjust the layout to prevent overlap
        plt.savefig('random_forest_feature_importance.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen
        self.models['Random Forest'] = model  # Store the trained model in the models dictionary
        joblib.dump(model, 'Randomforest_model.pkl') 
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]  # Get the predicted probabilities for the Fraud class
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)  # Calculate the false positive and true positive rates for the ROC curve
        self.model_metrics['Random Forest'] = {  # Store the model’s performance metrics
            'Accuracy': model.score(self.X_test, self.y_test),  # Calculate the accuracy on the test set
            'ROC AUC': auc(fpr, tpr)  # Calculate the area under the ROC curve
        }

    # This method trains a Decision Tree model to predict fraud
    def train_decision_tree(self):
        print("\n=== Training Decision Tree ===")  # Print a header to show we’re starting Decision Tree
        model = DecisionTreeClassifier(random_state=42)  # Create a Decision Tree model with a fixed random seed for reproducibility
        model.fit(self.X_train, self.y_train)  # Train the model on the training data (features and labels)
        y_pred = model.predict(self.X_test)  # Make predictions on the test data
        print("\nClassification Report:")  # Print a subheader for the classification report
        print(classification_report(self.y_test, y_pred))  # Print a detailed report with precision, recall, and F1-score for each class
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)  # Perform 5-fold cross-validation on the training data to check consistency
        print("\nCross-validation scores:", cv_scores)  # Print the scores for each fold
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")  # Print the average score and variability (2 standard deviations)
        self.plot_confusion_matrix(self.y_test, y_pred, "Decision Tree Confusion Matrix")  # Plot the confusion matrix to show prediction errors
        self.plot_roc_curve(model, "Decision Tree")  # Plot the ROC curve to evaluate the model’s performance
        self.models['Decision Tree'] = model  # Store the trained model in the models dictionary for later use
        joblib.dump(model, 'DecisionTree_model.pkl') 
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]  # Get the predicted probabilities for the Fraud class
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)  # Calculate the false positive and true positive rates for the ROC curve
        self.model_metrics['Decision Tree'] = {  # Store the model’s performance metrics in the model_metrics dictionary
            'Accuracy': model.score(self.X_test, self.y_test),  # Calculate the accuracy on the test set
            'ROC AUC': auc(fpr, tpr)  # Calculate the area under the ROC curve
        }

    # This method trains a K-Nearest Neighbors (KNN) model to predict fraud
    def train_knn(self):
        print("\n=== Training K-Nearest Neighbors ===")  # Print a header to show we’re starting KNN
        k_range = range(1, 20)  # Define a range of k values (number of neighbors) to test, from 1 to 19
        k_scores = []  # Create an empty list to store the cross-validation scores for each k
        # Loop through each k value to find the best one
        for k in k_range:
            model = KNeighborsClassifier(n_neighbors=k)  # Create a KNN model with the current k value
            scores = cross_val_score(model, self.X_train, self.y_train, cv=5)  # Perform 5-fold cross-validation on the training data
            k_scores.append(scores.mean())  # Store the average score for this k value
        plt.figure(figsize=(10, 6))  # Create a new figure with a size of 10x6 inches
        plt.plot(k_range, k_scores)  # Plot the cross-validation scores against the k values
        plt.xlabel('Value of K')  # Label the x-axis as the k values
        plt.ylabel('Cross-validated Accuracy')  # Label the y-axis as the accuracy scores
        plt.title('Optimal K Value Selection')  # Add a title to the plot
        plt.grid(True)  # Add a grid to make the plot easier to read
        plt.savefig('knn_k_selection.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen
        best_k = k_range[k_scores.index(max(k_scores))]  # Find the k value with the highest cross-validation score
        print(f"\nOptimal K value: {best_k}")  # Print the best k value we found
        model = KNeighborsClassifier(n_neighbors=best_k)  # Create a new KNN model with the best k value
        model.fit(self.X_train, self.y_train)  # Train the model on the training data
        y_pred = model.predict(self.X_test)  # Make predictions on the test data
        print("\nClassification Report:")  # Print a subheader for the classification report
        print(classification_report(self.y_test, y_pred))  # Print a detailed report with precision, recall, and F1-score
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)  # Perform 5-fold cross-validation with the best model
        print("\nCross-validation scores:", cv_scores)  # Print the scores for each fold
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")  # Print the average score and variability
        self.plot_confusion_matrix(self.y_test, y_pred, "KNN Confusion Matrix")  # Plot the confusion matrix to show prediction errors
        self.plot_roc_curve(model, "K-Nearest Neighbors")  # Plot the ROC curve to evaluate the model’s performance
        self.models['KNN'] = model  # Store the trained model in the models dictionary
        joblib.dump(model, 'KNN_model.pkl') 
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]  # Get the predicted probabilities for the Fraud class
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)  # Calculate the false positive and true positive rates for the ROC curve
        self.model_metrics['KNN'] = {  # Store the model’s performance metrics
            'Accuracy': model.score(self.X_test, self.y_test),  # Calculate the accuracy on the test set
            'ROC AUC': auc(fpr, tpr)  # Calculate the area under the ROC curve
        }

    # This method trains a Support Vector Machine (SVM) model to predict fraud
    def train_svm(self):
        print("\n=== Training Support Vector Machine ===")  # Print a header to show we’re starting SVM
        param_grid = {  # Define the hyperparameters we want to test for tuning the model
            'C': [0.1, 1, 10],  # Regularisation parameter (try 0.1, 1, or 10)
            'gamma': ['scale', 'auto']  # Kernel parameter (try 'scale' or 'auto')
        }
        base_model = SVC(probability=True, random_state=42)  # Create an SVM model with probability estimates enabled and a fixed random seed
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)  # Set up a grid search to find the best hyperparameters, using 5-fold cross-validation
        grid_search.fit(self.X_train, self.y_train)  # Run the grid search on the training data
        model = grid_search.best_estimator_  # Get the best model from the grid search
        print(f"Best parameters: {grid_search.best_params_}")  # Print the best hyperparameters found
        y_pred = model.predict(self.X_test)  # Make predictions on the test data
        print("\nClassification Report:")  # Print a subheader for the classification report
        print(classification_report(self.y_test, y_pred))  # Print a detailed report with precision, recall, and F1-score
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)  # Perform 5-fold cross-validation with the best model
        print("\nCross-validation scores:", cv_scores)  # Print the scores for each fold
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")  # Print the average score and variability
        self.plot_confusion_matrix(self.y_test, y_pred, "SVM Confusion Matrix")  # Plot the confusion matrix to show prediction errors
        self.plot_roc_curve(model, "Support Vector Machine")  # Plot the ROC curve to evaluate the model’s performance
        self.models['SVM'] = model  # Store the trained model in the models dictionary
        joblib.dump(model, 'SVM_model.pkl') 
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]  # Get the predicted probabilities for the Fraud class
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)  # Calculate the false positive and true positive rates for the ROC curve
        self.model_metrics['SVM'] = {  # Store the model’s performance metrics
            'Accuracy': model.score(self.X_test, self.y_test),  # Calculate the accuracy on the test set
            'ROC AUC': auc(fpr, tpr)  # Calculate the area under the ROC curve
        }

    # This method trains an XGBoost model to predict fraud
    def train_xgboost(self):
        print("\n=== Training XGBoost ===")  # Print a header to show we’re starting XGBoost
        param_grid = {  # Define the hyperparameters we want to test for tuning the model
            'n_estimators': [50, 100, 200],  # Number of boosting rounds
            'max_depth': [3, 6, 9],  # Maximum depth of each tree
            'learning_rate': [0.01, 0.1, 0.3]  # Step size shrinkage to prevent overfitting
        }
        base_model = XGBClassifier(random_state=42, eval_metric='logloss', )  # Create an XGBoost model with a fixed random seed
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)  # Set up a grid search to find the best hyperparameters, using 5-fold cross-validation
        grid_search.fit(self.X_train, self.y_train)  # Run the grid search on the training data
        model = grid_search.best_estimator_  # Get the best model from the grid search
        print(f"Best parameters: {grid_search.best_params_}")  # Print the best hyperparameters found
        y_pred = model.predict(self.X_test)  # Make predictions on the test data
        print("\nClassification Report:")  # Print a subheader for the classification report
        print(classification_report(self.y_test, y_pred))  # Print a detailed report with precision, recall, and F1-score for each class
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)  # Perform 5-fold cross-validation with the best model
        print("\nCross-validation scores:", cv_scores)  # Print the scores for each fold
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")  # Print the average score and variability
        self.plot_confusion_matrix(self.y_test, y_pred, "XGBoost Confusion Matrix")  # Plot the confusion matrix to show prediction errors
        self.plot_roc_curve(model, "XGBoost")  # Plot the ROC curve to evaluate the model’s performance
        feature_names = pd.get_dummies(self.df_cleaned.drop('Fraud', axis=1)).columns  # Get the names of the features after converting categorical variables
        # Create a DataFrame with the feature names and their importance scores
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)  # Sort by importance (highest to lowest)
        plt.figure(figsize=(10, 6))  # Create a new figure with a size of 10x6 inches
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')  # Plot a bar chart of the top 10 most important features
        plt.title('Top 10 Most Important Features (XGBoost)')  # Add a title to the plot
        plt.xlabel('Feature Importance')  # Label the x-axis
        plt.tight_layout()  # Adjust the layout to prevent overlap
        plt.savefig('xgboost_feature_importance.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen
        self.models['XGBoost'] = model  # Store the trained model in the models dictionary
        joblib.dump(model, 'XGBoost_model.pkl') 
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]  # Get the predicted probabilities for the Fraud class
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)  # Calculate the false positive and true positive rates for the ROC curve
        self.model_metrics['XGBoost'] = {  # Store the model’s performance metrics
            'Accuracy': model.score(self.X_test, self.y_test),  # Calculate the accuracy on the test set
            'ROC AUC': auc(fpr, tpr)  # Calculate the area under the ROC curve
        }

    # This method compares the performance of all trained supervised models
    def compare_supervised_models(self):
        if not self.model_metrics:  # Check if we’ve trained any models yet
            print("\nPlease train at least one supervised learning model first (Options 5-10)")  # Print a message if no models have been trained
            return  # Exit the method if there are no models to compare
        print("\n=== Supervised Learning Model Comparison ===")  # Print a header for the comparison
        comparison_df = pd.DataFrame(self.model_metrics).T  # Convert the model_metrics dictionary to a DataFrame (transpose so models are rows)
        print("\nModel Performance Metrics:")  # Print a subheader for the metrics
        print(comparison_df)  # Print the DataFrame showing accuracy and ROC AUC for each model
        comparison_df.to_csv('supervised_model_comparison.csv', index=True)  # Save the comparison to a CSV file
        plt.figure(figsize=(10, 6))  # Create a new figure with a size of 10x6 inches
        sns.barplot(x='Accuracy', y=comparison_df.index, data=comparison_df)  # Plot a bar chart comparing the accuracy of each model
        plt.title('Model Accuracy Comparison')  # Add a title to the plot
        plt.xlabel('Accuracy')  # Label the x-axis
        plt.savefig('supervised_accuracy_comparison.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen
        plt.figure(figsize=(10, 6))  # Create a new figure with a size of 10x6 inches
        sns.barplot(x='ROC AUC', y=comparison_df.index, data=comparison_df)  # Plot a bar chart comparing the ROC AUC of each model
        plt.title('Model ROC AUC Comparison')  # Add a title to the plot
        plt.xlabel('ROC AUC')  # Label the x-axis
        plt.savefig('supervised_roc_auc_comparison.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen

    def run_kmeans(self):
        print("\n=== Running K-Means Clustering (Elbow Method Only) ===")  # Print a header to show we’re starting K-Means with the Elbow Method only
        inertias = []  # Create an empty list to store inertia (how tightly the data points are clustered) for each k value
        k_range = range(2, 11)  # Define a range of k values (number of clusters) to test, from 2 to 10

        # Loop through each k value to calculate inertia
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)  # Create a K-Means model with the current k value and fixed seed for consistency
            kmeans.fit(self.unsupervised_features)  # Fit the model to the scaled data
            inertias.append(kmeans.inertia_)  # Store the inertia for this k value

        # Plot the Elbow curve to help decide the best number of clusters
        plt.figure(figsize=(8, 6))  # Set the figure size for clarity
        plt.plot(k_range, inertias, 'bo-')  # Plot the inertia values as a line with blue dots
        plt.xlabel('Number of Clusters (k)')  # Label the x-axis
        plt.ylabel('Inertia')  # Label the y-axis
        plt.title('Elbow Method For Optimal k')  # Add a title to the plot
        plt.grid(True)  # Add a grid to make the plot easier to read
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig('kmeans_elbow_method_only.png')  # Save the plot to a PNG file
        plt.show()  # Display the plot on screen
        print("Saved Elbow plot to 'kmeans_elbow_method_only.png'")  # Confirm the plot has been saved

        # Ask the user to choose the best number of clusters based on the Elbow plot
        try:
            optimal_k = int(input("\nEnter the optimal number of clusters based on the Elbow plot (e.g., 3): "))  # Ask the user for the best k
        except ValueError:
            print("Invalid input. Using k=3 by default.")  # If user input is not a number, use 3 by default
            optimal_k = 3  # Set a default value for k

        # Run final K-Means clustering using the chosen k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)  # Create a K-Means model with the chosen number of clusters
        clusters = kmeans.fit_predict(self.unsupervised_features)  # Fit the model and assign cluster labels to each data point

        # Plot a scatter plot of the first two features, coloured by their cluster
        plt.figure(figsize=(8, 6))  # Set the figure size for clarity
        scatter = plt.scatter(self.unsupervised_features[:, 0], self.unsupervised_features[:, 1],
                              c=clusters, cmap='viridis')  # Plot the first two features with colours showing clusters
        plt.title(f'K-Means Clustering (k={optimal_k})')  # Add a title with the chosen number of clusters
        plt.xlabel('Feature 1')  # Label the x-axis
        plt.ylabel('Feature 2')  # Label the y-axis
        plt.colorbar(scatter)  # Add a colour bar to explain cluster colours
        plt.tight_layout()  # Adjust layout
        plt.savefig('kmeans_clusters_elbow_only.png')  # Save the clustering plot to a PNG file
        plt.show()  # Display the plot on screen
        print(f"Saved K-Means clustering plot for k={optimal_k} to 'kmeans_clusters_elbow_only.png'")  # Confirm the plot has been saved
        
        joblib.dump(kmeans, 'kmeans_model.pkl')

        # Store the clustering results for later comparison
        self.clustering_results['K-Means (Elbow Only)'] = {  # Save results under a custom label
            'Optimal k/eps': optimal_k,  # Save the chosen number of clusters
            'Number of Clusters': optimal_k  # Also save under this key for consistency
        }

    # This method runs K-Means++ clustering (a variation of K-Means with better initialisation)
    def run_kmeans_plus_plus(self):
        print("\n=== Running K-Means++ Clustering (Elbow Method Only) ===")  # Print a header to show we’re starting K-Means++
        inertias = []  # Create an empty list to store the inertia for each k
        k_range = range(2, 11)  # Define a range of k values to test, from 2 to 10
        # Loop through each k value to find the best number of clusters
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)  # Create a K-Means++ model with the current k value
            kmeans.fit(self.unsupervised_features)  # Fit the model to the scaled features
            inertias.append(kmeans.inertia_)  # Store the inertia for this k value
        plt.figure(figsize=(8, 6))  # Set the figure size for clarity
        plt.plot(k_range, inertias, 'bx-')  # Plot the inertia against k values
        plt.xlabel('k')  # Label the x-axis
        plt.ylabel('Inertia')  # Label the y-axis
        plt.title('Elbow Method For Optimal k (K-Means++)')  # Add a title to the elbow plot
        plt.grid(True)  # Add a grid to make the plot easier to read
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig('kmeans_plus_plus_elbow_only.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on screen
        print("Saved Elbow plot to 'kmeans_plus_plus_elbow_only.png'")  # Confirm the plot has been saved
        # Ask the user to choose the best number of clusters based on the Elbow plot
        try:
            optimal_k = int(input("\nEnter the optimal number of clusters based on the Elbow plot (e.g., 6): "))  # Ask the user for the best k
        except ValueError:
            print("Invalid input. Using k=6 by default.")  # If user input is not a number, use 6 by default
            optimal_k = 6  # Set a default value for k
        # Run final K-Means++ clustering using the chosen k
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)  # Create a new K-Means++ model with the optimal k value
        clusters = kmeans.fit_predict(self.unsupervised_features)  # Fit the model and predict the cluster labels
        plt.figure(figsize=(8, 6))  # Create a new figure with a size of 8x6 inches
        scatter = plt.scatter(self.unsupervised_features[:, 0], self.unsupervised_features[:, 1],
                              c=clusters, cmap='viridis')  # Plot a scatter plot of the first two features, coloured by cluster
        plt.title(f'K-Means++ Clustering (k={optimal_k})')  # Add a title to the plot
        plt.xlabel('Feature 1')  # Label the x-axis
        plt.ylabel('Feature 2')  # Label the y-axis
        plt.colorbar(scatter)  # Add a colour bar to show the cluster colours
        plt.savefig('kmeans_plus_plus_clustering.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen
        joblib.dump(kmeans, 'kmeans_++_model.pkl')
        self.clustering_results['K-Means++ (Elbow Only)'] = {  # Store the clustering results in the clustering_results dictionary
            'Optimal k/eps': optimal_k,
            'Number of Clusters': optimal_k
        }

    # This method runs DBSCAN clustering to find patterns in the data
    def run_dbscan(self):
        print("\n=== Running DBSCAN Clustering ===")  # Print a header to show we’re starting DBSCAN
        eps_range = np.arange(0.3, 1.1, 0.1)  # Define a range of eps values (distance threshold) to test, from 0.3 to 1.0 in steps of 0.1
        min_samples_range = range(3, 10)  # Define a range of min_samples values (minimum points to form a cluster) to test, from 3 to 9
        best_silhouette = -1  # initialise the best silhouette score as -1 (worst possible score)
        best_eps = None  # initialise the best eps value as None
        best_min_samples = None  # initialise the best min_samples value as None
        best_labels = None  # initialise the best cluster labels as None
        # Loop through each combination of eps and min_samples to find the best parameters
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # Create a DBSCAN model with the current eps and min_samples
                labels = dbscan.fit_predict(self.unsupervised_features)  # Fit the model and predict the cluster labels
                # Check if there’s more than one cluster (silhouette score requires at least 2 clusters)
                if len(np.unique(labels)) > 1 and -1 in labels:  # Ensure there are clusters and noise points (-1 labels)
                    silhouette = silhouette_score(self.unsupervised_features, labels)  # Calculate the silhouette score
                    if silhouette > best_silhouette:  # If this score is better than the previous best
                        best_silhouette = silhouette  # Update the best silhouette score
                        best_eps = eps  # Update the best eps value
                        best_min_samples = min_samples  # Update the best min_samples value
                        best_labels = labels  # Update the best cluster labels
        if best_labels is None:  # Check if we found any valid clustering
            print("No valid clustering found with DBSCAN.")  # Print a message if no valid clustering was found
            return  # Exit the method if no clustering was found
        print(f"\nOptimal eps: {best_eps}, Optimal min_samples: {best_min_samples}")  # Print the best parameters found
        print(f"Silhouette Score: {best_silhouette:.3f}")  # Print the best silhouette score
        num_clusters = len(np.unique(best_labels)) - 1  # Calculate the number of clusters (excluding noise points labeled as -1)
        print(f"Number of clusters (excluding noise): {num_clusters}")  # Print the number of clusters
        plt.figure(figsize=(8, 6))  # Create a new figure with a size of 8x6 inches
        scatter = plt.scatter(self.unsupervised_features[:, 0], self.unsupervised_features[:, 1],
                              c=best_labels, cmap='viridis')  # Plot a scatter plot of the first two features, coloured by cluster
        plt.title(f'DBSCAN Clustering (eps={best_eps:.1f}, min_samples={best_min_samples})')  # Add a title to the plot
        plt.xlabel('Feature 1')  # Label the x-axis
        plt.ylabel('Feature 2')  # Label the y-axis
        plt.colorbar(scatter)  # Add a colour bar to show the cluster colours
        plt.savefig('dbscan_clustering.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen
        joblib.dump(dbscan, 'dbscan_model.pkl')
        self.clustering_results['DBSCAN'] = {  # Store the clustering results in the clustering_results dictionary
            'Optimal k/eps': best_eps,
            'Silhouette Score': best_silhouette,
            'Number of Clusters': num_clusters
        }

    def compare_clustering_methods(self):
        if not self.clustering_results:  # Check if we’ve run any clustering methods yet
            print("\nPlease run at least one clustering method first (Options 11-13)")
            return
        print("\n=== Clustering Method Comparison ===")
        comparison_df = pd.DataFrame(self.clustering_results).T  # Convert clustering_results to DataFrame
        print("\nClustering Performance Metrics:")
        print(comparison_df)

        # Calculate stability by running each method multiple times
        stability = {}
        for method in ['K-Means (Elbow Only)', 'K-Means++ (Elbow Only)', 'DBSCAN']:
            cluster_counts = []
            for _ in range(5):  # Run 5 times for stability
                if method == 'K-Means (Elbow Only)':
                    kmeans = KMeans(n_clusters=3, random_state=np.random.randint(0, 100))
                    labels = kmeans.fit_predict(self.unsupervised_features)
                    cluster_counts.append(len(np.unique(labels)) - (1 if -1 in labels else 0))
                elif method == 'K-Means++ (Elbow Only)':
                    kmeans = KMeans(n_clusters=6, init='k-means++', random_state=np.random.randint(0, 100))
                    labels = kmeans.fit_predict(self.unsupervised_features)
                    cluster_counts.append(len(np.unique(labels)) - (1 if -1 in labels else 0))
                elif method == 'DBSCAN':
                    dbscan = DBSCAN(eps=0.7, min_samples=4)
                    labels = dbscan.fit_predict(self.unsupervised_features)
                    cluster_counts.append(len(np.unique(labels)) - (1 if -1 in labels else 0))
            stability[method] = np.std(cluster_counts) if cluster_counts else 0

        # Add stability to comparison DataFrame
        comparison_df['Stability (Std Dev)'] = [stability.get(method, 0) for method in comparison_df.index]

        # Save comparison to CSV
        comparison_df.to_csv('clustering_comparison.csv', index=True)

        # Plot number of clusters
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Number of Clusters', y=comparison_df.index, data=comparison_df)
        plt.title('Number of Clusters Comparison')
        plt.xlabel('Number of Clusters')
        plt.savefig('clustering_num_clusters_comparison.png')
        plt.show()

        # Plot stability
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Stability (Std Dev)', y=comparison_df.index, data=comparison_df)
        plt.title('Cluster Stability Comparison')
        plt.xlabel('Standard Deviation of Cluster Counts')
        plt.savefig('clustering_stability_comparison.png')
        plt.show()

    # Alias for compare_clustering_methods to match the menu option name
    def compare_unsupervised_models(self):
        self.compare_clustering_methods()

        # This method runs the main program loop, providing a menu for the user to interact with
    def main(self):
        while True:  # Start an infinite loop to keep the program running until the user chooses to exit
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal screen (cls for Windows, clear for Unix-based systems)
            print("=== Fraud Analysis System ===")  # Print the program title
            print("\nGeneral Operations:")  # Add a subtitle for general operations
            print("1. Load and Clean Data")  # Option to load and clean the dataset
            print("2. View Data Summary")  # Option to view a summary of the dataset
            print("3. Analyse Fraud Relationships")  # Option to analyse relationships between features and fraud
            print("4. Plot Correlation Matrix")  # Option to plot a correlation matrix for numerical features
            
            print("\nSupervised Learning:")  # Add a subtitle for supervised learning options
            print("5. Train Logistic Regression")  # Option to train a Logistic Regression model
            print("6. Train Random Forest")  # Option to train a Random Forest model
            print("7. Train Decision Tree")  # Option to train a Decision Tree model
            print("8. Train K-Nearest Neighbors")  # Option to train a K-Nearest Neighbors model
            print("9. Train Support Vector Machine")  # Option to train a Support Vector Machine model
            print("10. Train XGBoost")  # Option to train an XGBoost model
            print("11. Compare Supervised Models")  # Option to compare the performance of all supervised models
            
            print("\nUnsupervised Learning:")  # Add a subtitle for unsupervised learning options
            print("12. Run K-Means Clustering")  # Option to run K-Means clustering
            print("13. Run K-Means++ Clustering")  # Option to run K-Means++ clustering
            print("14. Run DBSCAN Clustering")  # Option to run DBSCAN clustering
            print("15. Compare Unsupervised Models")  # Option to compare the performance of all unsupervised models
            
            print("\nExit:")  # Add a subtitle for the exit option
            print("16. Exit")  # Option to exit the program
            
            choice = input("\nEnter your choice (1-16): ")  # Ask the user to enter their choice
            # Handle the user’s choice
            if choice == '1':
                file_path = input("\nEnter the path to your CSV file: ")  # Ask the user for the path to the CSV file
                try:
                    self.df_original = pd.read_csv(file_path)  # Try to load the CSV file into a DataFrame
                    self.df_cleaned = self.clean_data(self.df_original)  # Clean the dataset using the clean_data method
                    self.plot_comparisons(self.df_original, self.df_cleaned)  # Plot comparisons of the data before and after cleaning
                except Exception as e:
                    print(f"Error loading file: {e}")  # Print an error message if the file can’t be loaded
            elif choice == '2':
                if self.df_cleaned is None:  # Check if the dataset has been loaded and cleaned
                    print("\nPlease load and clean the data first (Option 1)")  # Print a message if the data isn’t ready
                else:
                    self.print_data_summary()  # Print a summary of the dataset
            elif choice == '3':
                if self.df_cleaned is None:  # Check if the dataset has been loaded and cleaned
                    print("\nPlease load and clean the data first (Option 1)")  # Print a message if the data isn’t ready
                else:
                    self.analyse_fraud_relationships(self.df_cleaned)  # Analyse relationships between features and fraud
            elif choice == '4':
                if self.df_cleaned is None:  # Check if the dataset has been loaded and cleaned
                    print("\nPlease load and clean the data first (Option 1)")  # Print a message if the data isn’t ready
                else:
                    self.plot_correlation_matrix(self.df_cleaned)  # Plot a correlation matrix for numerical features
            elif choice in ['5', '6', '7', '8', '9', '10']:  # Options for training supervised models
                if self.df_cleaned is None:  # Check if the dataset has been loaded and cleaned
                    print("\nPlease load and clean the data first (Option 1)")  # Print a message if the data isn’t ready
                else:
                    self.prepare_data(for_supervised=True)  # Prepare the data for supervised learning
                    if choice == '5':
                        self.train_logistic_regression()  # Train the Logistic Regression model
                    elif choice == '6':
                        self.train_random_forest()  # Train the Random Forest model
                    elif choice == '7':
                        self.train_decision_tree()  # Train the Decision Tree model
                    elif choice == '8':
                        self.train_knn()  # Train the K-Nearest Neighbors model
                    elif choice == '9':
                        self.train_svm()  # Train the Support Vector Machine model
                    elif choice == '10':
                        self.train_xgboost()  # Train the XGBoost model
            elif choice == '11':
                self.compare_supervised_models()  # Compare the performance of all supervised models
            elif choice in ['12', '13', '14']:  # Options for running unsupervised models
                if self.df_cleaned is None:  # Check if the dataset has been loaded and cleaned
                    print("\nPlease load and clean the data first (Option 1)")  # Print a message if the data isn’t ready
                else:
                    self.prepare_data(for_supervised=False)  # Prepare the data for unsupervised learning
                    if choice == '12':
                        self.run_kmeans()  # Run K-Means clustering
                    elif choice == '13':
                        self.run_kmeans_plus_plus()  # Run K-Means++ clustering
                    elif choice == '14':
                        self.run_dbscan()  # Run DBSCAN clustering
            elif choice == '15':
                self.compare_unsupervised_models()  # Compare the performance of all unsupervised models
            elif choice == '16':
                print("\nExiting Fraud Analysis System. Goodbye!")  # Print a goodbye message
                break  # Exit the loop to end the program
            else:
                print("\nInvalid choice. Please enter a number between 1 and 16.")  # Print a message if the user enters an invalid option
            input("\nPress Enter to continue...")  # Pause the program until the user presses Enter

# This block runs the program if the script is executed directly (not imported as a module)
if __name__ == "__main__":
    fraud_system = FraudAnalysisSystem()  # Create a new instance of the FraudAnalysisSystem class
    fraud_system.main()  # Run the main method to start the program