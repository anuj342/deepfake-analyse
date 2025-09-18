import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# --- CONFIGURATION ---
MODEL_PATH = 'deepfake_webapp/deepfake_detector_model.h5'  # Path to your saved model
VALIDATION_DATA_PATH = 'processed_data/validation/'
IMAGE_SIZE = 224
BATCH_SIZE = 32
# --- END CONFIGURATION ---

# 1. Load the trained model
print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# 2. Set up the validation data generator
# IMPORTANT: No data augmentation for validation/testing!
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DATA_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Very important to keep data in order for confusion matrix
)

# 3. Evaluate the model's performance
print("\n--- Evaluating Model ---")
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# 4. Generate predictions for the confusion matrix
print("\n--- Generating Predictions for Confusion Matrix ---")
# Get the true labels
y_true = validation_generator.classes

# Get the predicted probabilities
y_pred_probs = model.predict(validation_generator)
# Convert probabilities to class labels (0 or 1)
y_pred = (y_pred_probs > 0.5).astype(int).reshape(-1)

# 5. Create and display the confusion matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)

# Get class labels from the generator
class_labels = list(validation_generator.class_indices.keys())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# 6. Print the classification report
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_labels))