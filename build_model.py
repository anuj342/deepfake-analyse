import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION ---
# This must match the image size from your preprocessing step
IMAGE_SIZE = 224


# --- END CONFIGURATION ---

def build_deepfake_detector():
    """
    Builds the deepfake detection model using transfer learning with Xception.
    """
    print("Loading pre-trained Xception model...")
    # 1. Load the base Xception model, pre-trained on ImageNet.
    # We don't include the final classification layer ('include_top=False').
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # 2. Freeze the layers of the base model.
    # This prevents their pre-trained weights from being changed during the first phase of training.
    print("Freezing layers in the base model...")
    base_model.trainable = False

    # 3. Create our new custom classification head.
    # We'll add this on top of the base model.

    # Get the output of the base model
    x = base_model.output

    # Add a pooling layer to reduce dimensions
    x = GlobalAveragePooling2D()(x)

    # Add a fully-connected layer for high-level feature learning
    x = Dense(512, activation='relu')(x)

    # Add a dropout layer to prevent overfitting
    x = Dropout(0.5)(x)

    # Add the final output layer. A single neuron with a sigmoid activation
    # is perfect for binary (Real/Fake) classification.
    predictions = Dense(1, activation='sigmoid')(x)

    # 4. Assemble the final model.
    model = Model(inputs=base_model.input, outputs=predictions)

    print("Model assembled successfully.")
    return model


def compile_model(model):
    """
    Compiles the model with an optimizer, loss function, and metrics.
    """
    print("Compiling model...")
    # We use the Adam optimizer, a robust choice for most problems.
    # The loss function 'binary_crossentropy' is the standard for two-class problems.
    # We'll monitor 'accuracy' during training.
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("Compilation complete.")
    return model


# --- SCRIPT EXECUTION ---
if __name__ == '__main__':
    # Build the model
    deepfake_model = build_deepfake_detector()

    # Compile the model
    deepfake_model = compile_model(deepfake_model)

    # Print a summary of the model's architecture
    print("\n--- Model Summary ---")
    deepfake_model.summary()