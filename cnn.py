# Import required libraries for the project
import numpy as np  # This handles numerical calculations and arrays
import matplotlib.pyplot as plt  # This creates plots and graphs for visualisation
import seaborn as sns  # This enhances matplotlib plots, making them more visually appealing
import pandas as pd  # This manages data in tables (DataFrames) for analysis
import os  # This interacts with the operating system, e.g., for clearing the terminal
import cv2  # This handles image processing tasks (not used directly but imported for potential future use)
import time  # This tracks the time taken for training models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  # These evaluate model performance: confusion_matrix shows prediction errors, classification_report gives precision/recall, accuracy_score measures accuracy
from tensorflow.keras.models import Sequential  # This builds a sequential neural network model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization  # These are layers for the neural network: Conv2D for convolutions, MaxPooling2D for downsampling, Flatten for vector conversion, Dense for fully connected layers, Dropout for regularisation, BatchNormalization for normalising layer inputs
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # This handles image data loading and augmentation
from tensorflow.keras.callbacks import EarlyStopping, Callback  # EarlyStopping stops training early if performance stops improving, Callback is the base class for custom callbacks
from tensorflow.keras.optimizers import Adam  # This is the Adam optimiser for training the model

# Define a class to manage the Intel Image Classification experiment
class IntelImageClassifierExperiment:
    # Custom callback to stop training when validation accuracy reaches 100%
    class StopAtPerfectAccuracy(Callback):
        """Custom callback to stop training when validation accuracy reaches 100%"""
        def on_epoch_end(self, epoch, logs=None):
            val_acc = logs.get('val_accuracy')  # Get the validation accuracy from the logs
            if val_acc is not None and val_acc >= 1.0:  # Check if val_accuracy is 100% (1.0)
                print(f"\nValidation accuracy reached 100% at epoch {epoch + 1}. Stopping training.")
                self.model.stop_training = True  # Stop the training process immediately

    # This is the constructor method that runs when we create a new IntelImageClassifierExperiment object
    def __init__(self, data_dir):
        self.data_dir = data_dir  # Store the directory path where the image dataset is located
        self.class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']  # List the categories of images we’ll classify
        self.img_height = 150  # Set the height for resizing images to 150 pixels
        self.img_width = 150  # Set the width for resizing images to 150 pixels
        self.train_generator = None  # This will store the training data generator
        self.validation_generator = None  # This will store the validation data generator
        self.test_generator = None  # This will store the test data generator
        self.results_df = pd.DataFrame(columns=[
            'Model Type', 'Batch Size', 'Learning Rate', 'Dropout Rate', 
            'Conv Layers', 'Training Time (s)', 'Accuracy', 'Validation Accuracy'
        ])  # Create an empty DataFrame to store experiment results with columns for model details and performance metrics
    
    # This method loads and preprocesses the image data for training, validation, and testing
    def load_and_preprocess_data(self, batch_size=32):
        """Load and preprocess the data with the given batch size"""
        print(f"\nLoading data with batch size: {batch_size}")  # Print the batch size being used for loading data
        
        # Data augmentation for training to improve model robustness
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalise pixel values to the range [0, 1] by dividing by 255
            rotation_range=20,  # Randomly rotate images by up to 20 degrees
            width_shift_range=0.2,  # Randomly shift images horizontally by up to 20% of their width
            height_shift_range=0.2,  # Randomly shift images vertically by up to 20% of their height
            shear_range=0.2,  # Apply random shear transformations by up to 20%
            zoom_range=0.2,  # Randomly zoom into images by up to 20%
            horizontal_flip=True,  # Randomly flip images horizontally
            validation_split=0.2  # Reserve 20% of the training data for validation
        )

        # Only rescaling for testing to keep test data unmodified except for normalisation
        test_datagen = ImageDataGenerator(rescale=1./255)  # Normalise test images by dividing pixel values by 255
        
        # Load training data with validation split
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'seg_train'),  # Path to the training data directory
            target_size=(self.img_height, self.img_width),  # Resize images to 150x150 pixels
            batch_size=batch_size,  # Set the batch size for training
            class_mode='sparse',  # Use sparse labels (integers) for classification
            subset='training',  # Use the training subset (80% of seg_train)
            shuffle=True  # Shuffle the training data to improve learning
        )
        
        self.validation_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'seg_train'),  # Path to the training data directory (for validation split)
            target_size=(self.img_height, self.img_width),  # Resize images to 150x150 pixels
            batch_size=batch_size,  # Set the batch size for validation
            class_mode='sparse',  # Use sparse labels (integers) for classification
            subset='validation',  # Use the validation subset (20% of seg_train)
            shuffle=False  # Don’t shuffle validation data to maintain consistency
        )
        
        # Load test data
        self.test_generator = test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'seg_test'),  # Path to the test data directory
            target_size=(self.img_height, self.img_width),  # Resize images to 150x150 pixels
            batch_size=batch_size,  # Set the batch size for testing
            class_mode='sparse',  # Use sparse labels (integers) for classification
            shuffle=False  # Don’t shuffle test data to maintain order for evaluation
        )
        
        # Print the number of images loaded for each set
        print(f"Found {self.train_generator.samples} training images")  # Display the number of training images
        print(f"Found {self.validation_generator.samples} validation images")  # Display the number of validation images
        print(f"Found {self.test_generator.samples} test images")  # Display the number of test images
    
    # This method builds a Convolutional Neural Network (CNN) model with specified parameters
    def build_cnn_model(self, learning_rate=0.001, dropout_rate=0.5, conv_layers=3):
        """Build a CNN model with specified parameters"""
        model = Sequential()  # Create a sequential model where layers are added one after another
        
        # First convolutional block to extract basic features from images
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=(self.img_height, self.img_width, 3)))  # Add a convolutional layer with 32 filters, 3x3 kernel, ReLU activation, and same padding
        model.add(BatchNormalization())  # Normalise the layer outputs to stabilise training
        model.add(MaxPooling2D(2, 2))  # Downsample the feature maps by taking the maximum value in each 2x2 region
        
        # Additional convolutional blocks based on the number of conv_layers specified
        filters = 64  # Start with 64 filters for the next layer
        for _ in range(conv_layers - 1):  # Loop to add additional convolutional blocks
            model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))  # Add a convolutional layer with increasing filters
            model.add(BatchNormalization())  # Normalise the layer outputs
            model.add(MaxPooling2D(2, 2))  # Downsample the feature maps
            filters *= 2  # Double the number of filters for the next layer
            if filters > 256:  # Cap the number of filters at 256 to avoid excessive growth
                filters = 256
        
        # Flatten and dense layers to process the extracted features
        model.add(Flatten())  # Flatten the feature maps into a 1D vector for dense layers
        model.add(Dense(256, activation='relu'))  # Add a dense layer with 256 units and ReLU activation
        model.add(BatchNormalization())  # Normalise the layer outputs
        model.add(Dropout(dropout_rate))  # Add dropout to prevent overfitting by randomly dropping units
        model.add(Dense(128, activation='relu'))  # Add another dense layer with 128 units
        model.add(Dropout(dropout_rate))  # Add another dropout layer
        model.add(Dense(len(self.class_names), activation='softmax'))  # Add the output layer with units equal to the number of classes, using softmax for probabilities
        
        # Compile the model for training
        optimizer = Adam(learning_rate=learning_rate)  # Create an Adam optimiser with the specified learning rate
        model.compile(
            optimizer=optimizer,  # Use the Adam optimiser
            loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy as the loss function for multi-class classification
            metrics=['accuracy']  # Track accuracy during training
        )
        
        return model  # Return the compiled CNN model
    
    # This method builds a Multi-Layer Perceptron (MLP) model with specified parameters
    def build_mlp_model(self, learning_rate=0.001, dropout_rate=0.5):
        """Build an MLP model with specified parameters"""
        model = Sequential([  # Create a sequential model with the following layers
            Flatten(input_shape=(self.img_height, self.img_width, 3)),  # Flatten the input images into a 1D vector
            Dense(512, activation='relu'),  # Add a dense layer with 512 units and ReLU activation
            BatchNormalization(),  # Normalise the layer outputs to stabilise training
            Dropout(dropout_rate),  # Add dropout to prevent overfitting
            Dense(256, activation='relu'),  # Add a dense layer with 256 units
            BatchNormalization(),  # Normalise the layer outputs
            Dropout(dropout_rate),  # Add another dropout layer
            Dense(128, activation='relu'),  # Add a dense layer with 128 units
            Dropout(dropout_rate),  # Add another dropout layer
            Dense(len(self.class_names), activation='softmax')  # Add the output layer with softmax activation for classification
        ])
        
        # Compile the model for training
        optimizer = Adam(learning_rate=learning_rate)  # Create an Adam optimiser with the specified learning rate
        model.compile(
            optimizer=optimizer,  # Use the Adam optimiser
            loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy as the loss function
            metrics=['accuracy']  # Track accuracy during training
        )
        
        return model  # Return the compiled MLP model
    
    # This method trains and evaluates a model with the specified parameters
    def train_and_evaluate_model(self, model_type, batch_size, learning_rate, dropout_rate, conv_layers=3, epochs=10):
        """Train and evaluate a model with the specified parameters"""
        # Load data with the current batch size
        self.load_and_preprocess_data(batch_size)  # Call the method to load and preprocess the data
        
        # Build the appropriate model based on the model type
        if model_type == 'CNN':
            model = self.build_cnn_model(learning_rate, dropout_rate, conv_layers)  # Build a CNN model with the given parameters
            model_name = f"CNN (Conv Layers: {conv_layers})"  # Create a name for the model including the number of conv layers
        else:  # MLP
            model = self.build_mlp_model(learning_rate, dropout_rate)  # Build an MLP model with the given parameters
            model_name = "MLP"  # Set the model name to MLP
        
        # Early stopping to prevent overfitting and stop if validation accuracy doesn't improve
        early_stopping = EarlyStopping(
            monitor='val_accuracy',  # Monitor the validation accuracy instead of validation loss
            min_delta=0,  # Require any improvement in validation accuracy (no minimum threshold)
            patience=3,  # Stop after 3 epochs with no improvement in validation accuracy
            restore_best_weights=True  # Restore the model weights from the epoch with the best validation accuracy
        )
        
        # Custom callback to stop training immediately if validation accuracy reaches 100%
        stop_at_perfect = self.StopAtPerfectAccuracy()  # Create an instance of the custom callback
        
        # Train the model and measure the time taken
        start_time = time.time()  # Record the start time of training
        history = model.fit(
            self.train_generator,  # Use the training data generator
            epochs=epochs,  # Train for the specified number of epochs
            validation_data=self.validation_generator,  # Use the validation data for monitoring
            callbacks=[early_stopping, stop_at_perfect],  # Apply both early stopping and the custom callback
            verbose=1  # Show training progress
        )
        train_time = time.time() - start_time  # Calculate the total training time
        
        # Evaluate the model on the training and validation sets
        _, train_acc = model.evaluate(self.train_generator, verbose=0)  # Get the training accuracy
        _, val_acc = model.evaluate(self.validation_generator, verbose=0)  # Get the validation accuracy
        
        # Create a new DataFrame row with the current results
        new_row = pd.DataFrame([{
            'Model Type': model_name,  # Store the model name
            'Batch Size': batch_size,  # Store the batch size used
            'Learning Rate': learning_rate,  # Store the learning rate used
            'Dropout Rate': dropout_rate,  # Store the dropout rate used
            'Conv Layers': conv_layers if model_type == 'CNN' else 'N/A',  # Store the number of conv layers (or N/A for MLP)
            'Training Time (s)': round(train_time, 2),  # Store the training time rounded to 2 decimal places
            'Accuracy': round(train_acc, 4),  # Store the training accuracy rounded to 4 decimal places
            'Validation Accuracy': round(val_acc, 4)  # Store the validation accuracy rounded to 4 decimal places
        }])
        
        # Add the results to the DataFrame using pd.concat
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)  # Append the new row to the existing DataFrame
        
        # Generate a confusion matrix for this model using the test set
        y_pred = np.argmax(model.predict(self.test_generator), axis=1)  # Predict the classes for the test set
        cm = confusion_matrix(self.test_generator.classes[:len(y_pred)], y_pred)  # Create a confusion matrix comparing true and predicted labels
        
        # Plot and save the confusion matrix
        plt.figure(figsize=(10, 8))  # Create a new figure with a size of 10x8 inches
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)  # Plot the confusion matrix as a heatmap with labels
        plt.xlabel("Predicted")  # Label the x-axis as predicted classes
        plt.ylabel("True")  # Label the y-axis as true classes
        plt.title(f"{model_name} Confusion Matrix (Batch: {batch_size}, LR: {learning_rate})")  # Add a title with the model details
        plt.tight_layout()  # Adjust the layout to prevent overlap
        plt.savefig(f"confusion_matrix_{model_type}_{batch_size}_{str(learning_rate).replace('.', '')}.png")  # Save the plot as a PNG file
        plt.close()  # Close the plot to free memory
        
        # Return the training history, model, and performance metrics
        return history, model, train_time, train_acc, val_acc
    
    # This method runs a grid search to find the best hyperparameter combination
    def run_parameter_experiments(self):
        """Run a grid search to find the best hyperparameter combination"""
        print("\n=== Starting Grid Search for Hyperparameter Tuning ===")  # Print a header to indicate the start of the grid search
        
        # Define the hyperparameter grid for tuning
        model_types = ['CNN', 'MLP']  # Define the model types to test: CNN and MLP
        batch_sizes = [16, 32, 64, 128]  # Define a range of batch sizes to test: 16, 32, 64, 128
        learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]  # Define a range of learning rates to test: 0.01, 0.005, 0.001, 0.0005, 0.0001
        dropout_rates = [0.2, 0.3, 0.4, 0.5, 0.6]  # Define a range of dropout rates to test: 0.2, 0.3, 0.4, 0.5, 0.6
        conv_layers = [2, 3, 4, 5]  # Define the number of convolutional layers to test for CNN: 2, 3, 4, 5
        
        # Log the best validation accuracy and configuration
        best_val_acc = 0  # Initialise the best validation accuracy as 0
        best_config = {}  # Create a dictionary to store the best configuration
        
        # Total number of combinations to try
        total_combinations = (len(batch_sizes) * len(learning_rates) * len(dropout_rates) * len(conv_layers) * (len(model_types) - 1)) + (len(batch_sizes) * len(learning_rates) * len(dropout_rates))  # Calculate the total number of combinations (CNN + MLP)
        print(f"Total combinations to try: {total_combinations}")  # Print the total number of combinations
        
        # Counter for tracking progress
        current_combination = 0  # Initialise a counter to track the current combination
        
        # Run grid search for CNN models
        for model_type in model_types:  # Loop through each model type (CNN and MLP)
            if model_type == 'CNN':  # Check if the model type is CNN
                for batch_size in batch_sizes:  # Loop through each batch size
                    for learning_rate in learning_rates:  # Loop through each learning rate
                        for dropout_rate in dropout_rates:  # Loop through each dropout rate
                            for conv_layer in conv_layers:  # Loop through each number of conv layers
                                current_combination += 1  # Increment the combination counter
                                print(f"\n--- Combination {current_combination}/{total_combinations}: Training {model_type} (Batch: {batch_size}, LR: {learning_rate}, Dropout: {dropout_rate}, Conv Layers: {conv_layer}) ---")  # Print the current experiment details
                                history, model, train_time, train_acc, val_acc = self.train_and_evaluate_model(
                                    model_type, batch_size, learning_rate, dropout_rate, conv_layer
                                )  # Train and evaluate the model with the current parameters
                                
                                # Check if this is the best model so far
                                if val_acc > best_val_acc:  # Compare the validation accuracy with the current best
                                    best_val_acc = val_acc  # Update the best validation accuracy
                                    best_config = {
                                        'Model Type': model_type,  # Store the model type
                                        'Batch Size': batch_size,  # Store the batch size
                                        'Learning Rate': learning_rate,  # Store the learning rate
                                        'Dropout Rate': dropout_rate,  # Store the dropout rate
                                        'Conv Layers': conv_layer,  # Store the number of conv layers
                                        'Validation Accuracy': val_acc  # Store the validation accuracy
                                    }
            else:  # MLP model
                for batch_size in batch_sizes:  # Loop through each batch size
                    for learning_rate in learning_rates:  # Loop through each learning rate
                        for dropout_rate in dropout_rates:  # Loop through each dropout rate
                            current_combination += 1  # Increment the combination counter
                            print(f"\n--- Combination {current_combination}/{total_combinations}: Training {model_type} (Batch: {batch_size}, LR: {learning_rate}, Dropout: {dropout_rate}) ---")  # Print the current experiment details
                            history, model, train_time, train_acc, val_acc = self.train_and_evaluate_model(
                                model_type, batch_size, learning_rate, dropout_rate
                            )  # Train and evaluate the MLP model with the current parameters
                            
                            # Check if this is the best model so far
                            if val_acc > best_val_acc:  # Compare the validation accuracy with the current best
                                best_val_acc = val_acc  # Update the best validation accuracy
                                best_config = {
                                    'Model Type': model_type,  # Store the model type
                                    'Batch Size': batch_size,  # Store the batch size
                                    'Learning Rate': learning_rate,  # Store the learning rate
                                    'Dropout Rate': dropout_rate,  # Store the dropout rate
                                    'Conv Layers': 'N/A',  # Conv layers don’t apply to MLP
                                    'Validation Accuracy': val_acc  # Store the validation accuracy
                                }
        
        # Display and save the experiment results
        print("\n=== Grid Search Results ===")  # Print a header for the results
        print(self.results_df.sort_values(by='Validation Accuracy', ascending=False))  # Print the results sorted by validation accuracy (highest first)
        self.results_df.to_csv('intel_image_classification_results.csv', index=False)  # Save the results to a CSV file
        
        # Display the best configuration found
        print("\n=== Best Configuration ===")  # Print a header for the best configuration
        for key, value in best_config.items():  # Loop through the best configuration details
            print(f"{key}: {value}")  # Print each key-value pair
        
        # Create a bar chart comparing validation accuracies across all models
        plt.figure(figsize=(12, 8))  # Create a new figure with a size of 12x8 inches
        models = self.results_df['Model Type'] + " (BS:" + self.results_df['Batch Size'].astype(str) + \
                ", LR:" + self.results_df['Learning Rate'].astype(str) + ")"  # Create labels for each model configuration
        plt.bar(models, self.results_df['Validation Accuracy'], color='skyblue')  # Plot a bar chart of validation accuracies
        plt.xlabel('Model Configuration')  # Label the x-axis
        plt.ylabel('Validation Accuracy')  # Label the y-axis
        plt.title('Comparison of Model Configurations')  # Add a title to the plot
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust the layout to prevent overlap
        plt.savefig('model_comparison.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen
        
        # Return the best configuration for further use
        return best_config
    
    # This method trains the final model with the best parameters for a longer duration
    def train_final_model(self, model_type, batch_size, learning_rate, dropout_rate, conv_layers=3, epochs=15):
        """Train the final model with the best parameters for longer"""
        print("\n=== Training Final Model with Best Configuration ===")  # Print a header to indicate the start of final training
        
        # Load data with the best batch size
        self.load_and_preprocess_data(batch_size)  # Call the method to load and preprocess the data
        
        # Build the appropriate model based on the model type
        if model_type == 'CNN':
            model = self.build_cnn_model(learning_rate, dropout_rate, conv_layers)  # Build a CNN model with the best parameters
            model_name = f"CNN (Conv Layers: {conv_layers})"  # Create a name for the model
        else:  # MLP
            model = self.build_mlp_model(learning_rate, dropout_rate)  # Build an MLP model with the best parameters
            model_name = "MLP"  # Set the model name to MLP
        
        # Train the model for the specified number of epochs
        history = model.fit(
            self.train_generator,  # Use the training data generator
            epochs=epochs,  # Train for the specified number of epochs
            validation_data=self.validation_generator,  # Use the validation data for monitoring
            verbose=1  # Show training progress
        )
        
        # Plot the training history (accuracy and loss)
        plt.figure(figsize=(12, 5))  # Create a new figure with a size of 12x5 inches
        
        # Plot accuracy over epochs
        plt.subplot(1, 2, 1)  # Create a subplot for accuracy (1 row, 2 columns, position 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')  # Plot the training accuracy
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Plot the validation accuracy
        plt.title('Model Accuracy')  # Add a title to the subplot
        plt.xlabel('Epoch')  # Label the x-axis
        plt.ylabel('Accuracy')  # Label the y-axis
        plt.legend()  # Show the legend
        
        # Plot loss over epochs
        plt.subplot(1, 2, 2)  # Create a subplot for loss (1 row, 2 columns, position 2)
        plt.plot(history.history['loss'], label='Training Loss')  # Plot the training loss
        plt.plot(history.history['val_loss'], label='Validation Loss')  # Plot the validation loss
        plt.title('Model Loss')  # Add a title to the subplot
        plt.xlabel('Epoch')  # Label the x-axis
        plt.ylabel('Loss')  # Label the y-axis
        plt.legend()  # Show the legend
        
        plt.tight_layout()  # Adjust the layout to prevent overlap
        plt.savefig('final_model_training.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen
        
        # Evaluate the model on the test set
        test_loss, test_acc = model.evaluate(self.test_generator)  # Get the test loss and accuracy
        print(f"\nFinal model test accuracy: {test_acc:.4f}")  # Print the test accuracy
        
        # Generate and plot a confusion matrix for the final model
        y_pred = np.argmax(model.predict(self.test_generator), axis=1)  # Predict the classes for the test set
        cm = confusion_matrix(self.test_generator.classes[:len(y_pred)], y_pred)  # Create a confusion matrix
        
        plt.figure(figsize=(10, 8))  # Create a new figure with a size of 10x8 inches
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)  # Plot the confusion matrix as a heatmap
        plt.xlabel("Predicted")  # Label the x-axis
        plt.ylabel("True")  # Label the y-axis
        plt.title(f"Final Model Confusion Matrix")  # Add a title to the plot
        plt.tight_layout()  # Adjust the layout
        plt.savefig('final_model_confusion_matrix.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen
        
        # Generate a classification report for the final model
        print("\nClassification Report:")  # Print a header for the classification report
        cls_report = classification_report(
            self.test_generator.classes[:len(y_pred)],
            y_pred,
            target_names=self.class_names,
            output_dict=True
        )  # Create a classification report with precision, recall, and F1-score for each class
        cls_report_df = pd.DataFrame(cls_report).transpose()  # Convert the report to a DataFrame
        print(cls_report_df)  # Print the classification report
        
        # Save the classification report to a CSV file
        cls_report_df.to_csv('final_model_classification_report.csv')  # Save the report as a CSV file
        
        # Display some example predictions to visualise the model’s performance
        self.display_example_predictions(model, 10)  # Call the method to display example predictions
        
        return model, test_acc  # Return the trained model and test accuracy
    
    # This method displays examples of the model’s predictions on test images
    def display_example_predictions(self, model, num_examples=5):
        """Display examples of model predictions"""
        # Get a batch of test images and their labels
        batch_images, batch_labels = next(self.test_generator)  # Fetch the next batch from the test generator
        
        # Make predictions on the batch of images
        predictions = model.predict(batch_images)  # Predict the probabilities for each class
        pred_classes = np.argmax(predictions, axis=1)  # Get the predicted class for each image
        
        # Display the predictions alongside the true labels
        plt.figure(figsize=(15, num_examples * 3))  # Create a new figure with a size based on the number of examples
        for i in range(min(num_examples, len(batch_images))):  # Loop through the specified number of examples
            plt.subplot(num_examples, 2, i*2 + 1)  # Create a subplot for the image (left side)
            plt.imshow(batch_images[i])  # Display the image
            plt.title(f"True: {self.class_names[int(batch_labels[i])]})")  # Show the true label
            plt.axis('off')  # Hide the axes for a cleaner look
            
            # Display the prediction probabilities as a bar chart
            plt.subplot(num_examples, 2, i*2 + 2)  # Create a subplot for the probabilities (right side)
            plt.barh(self.class_names, predictions[i])  # Plot a horizontal bar chart of probabilities
            plt.title(f"Prediction: {self.class_names[pred_classes[i]]}")  # Show the predicted label
        
        plt.tight_layout()  # Adjust the layout to prevent overlap
        plt.savefig('example_predictions.png')  # Save the plot as a PNG file
        plt.show()  # Display the plot on the screen

# This function runs the main program loop, providing a menu for the user to interact with
def main():
    # Set the path to the dataset directory (update this to your actual path)
    data_dir = r"C:\xampp\htdocs\xampp\heuristics\intel_image_classification"  # Define the path to the Intel Image Classification dataset
    
    # Create an instance of the IntelImageClassifierExperiment class
    experiment = IntelImageClassifierExperiment(data_dir)  # Initialise the experiment with the dataset path
    
    # Main menu loop to interact with the user
    while True:  # Start an infinite loop to keep the program running until the user exits
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal screen (cls for Windows, clear for Unix-based systems)
        print("\n=== Intel Image Classification Parameter Experiments ===")  # Print the program title
        print("\nOptions:")  # Add a subtitle for the menu options
        print("1. Run All Parameter Experiments (This will take time)")  # Option to run all experiments
        print("2. Train Final Model with Best Parameters")  # Option to train the final model
        print("3. Display Results Table")  # Option to display the results table
        print("4. Exit")  # Option to exit the program
        
        choice = input("\nEnter your choice (1-4): ")  # Ask the user to enter their choice
        
        if choice == '1':
            # Run parameter experiments with grid search
            best_config = experiment.run_parameter_experiments()  # Call the method to run the grid search
            input("\nExperiments complete. Press Enter to continue...")  # Wait for user input to continue
            
        elif choice == '2':
            # Check if experiments have been run to get the best parameters
            if experiment.results_df.empty:  # Check if the results DataFrame is empty
                print("\nPlease run parameter experiments first (Option 1)")  # Print a message if no experiments have been run
                input("\nPress Enter to continue...")  # Wait for user input
                continue  # Return to the menu
            
            # Get the best configuration from the results
            best_row = experiment.results_df.loc[experiment.results_df['Validation Accuracy'].idxmax()]  # Find the row with the highest validation accuracy
            
            # Train the final model with the best parameters
            model_type = 'CNN' if 'CNN' in best_row['Model Type'] else 'MLP'  # Determine the model type
            conv_layers = int(best_row['Conv Layers']) if model_type == 'CNN' else 3  # Get the number of conv layers (default to 3 if MLP)
            
            experiment.train_final_model(
                model_type=model_type,  # Use the best model type
                batch_size=int(best_row['Batch Size']),  # Use the best batch size
                learning_rate=float(best_row['Learning Rate']),  # Use the best learning rate
                dropout_rate=float(best_row['Dropout Rate']),  # Use the best dropout rate
                conv_layers=conv_layers,  # Use the best number of conv layers
                epochs=15  # Train for 15 epochs
            )
            
            input("\nFinal model training complete. Press Enter to continue...")  # Wait for user input to continue
            
        elif choice == '3':
            # Display the results table from the experiments
            if experiment.results_df.empty:  # Check if the results DataFrame is empty
                print("\nNo results available. Please run parameter experiments first (Option 1)")  # Print a message if no results exist
            else:
                print("\n=== Model Comparison Results ===")  # Print a header for the results
                print(experiment.results_df.sort_values(by='Validation Accuracy', ascending=False))  # Print the results sorted by validation accuracy
            
            input("\nPress Enter to continue...")  # Wait for user input to continue
            
        elif choice == '4':
            print("\nExiting the system...")  # Print a message to indicate the program is exiting
            break  # Exit the loop and end the program
        
        else:
            print("\nInvalid choice! Please try again.")  # Print a message if the user enters an invalid choice
            input("\nPress Enter to continue...")  # Wait for user input to continue

# This block runs the program if the script is executed directly (not imported as a module)
if __name__ == "__main__":
    main()  # Call the main function to start the program