import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION ---
TRAIN_DATA_PATH = 'processed_data/train/'  # Path to your training data
IMAGE_SIZE = 224  # Must match the size used in preprocessing and model building
BATCH_SIZE = 32  # Number of images to process in each batch
EPOCHS = 10  # Number of times to go through the entire dataset


# --- END CONFIGURATION ---

# --- MODEL BUILDING (Same as Step 2) ---
def build_deepfake_detector(image_size):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    base_model.trainable = False  # Freeze the base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- TRAINING SETUP ---
def get_train_generator(data_path, image_size, batch_size):
    print("Setting up training data generator...")
    # Create an ImageDataGenerator with data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # Rescale pixel values from 0-255 to 0-1
        rotation_range=15,  # Randomly rotate images
        width_shift_range=0.1,  # Randomly shift images horizontally
        height_shift_range=0.1,  # Randomly shift images vertically
        horizontal_flip=True,  # Randomly flip images horizontally
        zoom_range=0.1,  # Randomly zoom in on images
        fill_mode='nearest'
    )

    # The generator will load images from subdirectories and label them automatically
    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary'  # Because we have two classes: 'real' and 'fake'
    )
    print("Generator setup complete.")
    return train_generator


# --- SCRIPT EXECUTION ---
if __name__ == '__main__':
    # 1. Build and compile the model
    model = build_deepfake_detector(IMAGE_SIZE)
    model = compile_model(model)
    model.summary()

    # 2. Set up the data generator
    train_generator = get_train_generator(TRAIN_DATA_PATH, IMAGE_SIZE, BATCH_SIZE)

    # 3. Start training!
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        # Calculate steps_per_epoch to cover all data
        steps_per_epoch=train_generator.samples // BATCH_SIZE
    )
    print("--- Training Finished ---")

    # 4. Save the trained model for later use
    model.save('deepfake_detector_model.h5')
    print("Model saved to deepfake_detector_model.h5")