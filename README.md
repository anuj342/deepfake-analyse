# Deepfake video analyser

## Objective: To build a complete system that can analyze a video file and classify it as either "Real" or a "Deepfake." The project leverages deep learning to identify the subtle digital artifacts and inconsistencies that manipulated videos often contain.

## Requirements: 
1. Dataset used: Celeb-DF v1. Contains real and synthesized videos of celebrities interviews. It is available on kaggle (https://www.kaggle.com/datasets/reubensuju/celeb-df-v2).
2. Python packages: Tensorflow, Flask, OpenCV, MTCNN, Numpy, Scikit-learn.

## How to use:
### Folder structure:
<img width="571" height="348" alt="image" src="https://github.com/user-attachments/assets/37cf694b-17f1-4dda-90fd-c49ae29b5be4" />

### Process
1. Data Collection: Extract the downloaded dataset into the main project folder
2. Preprocessing: Prepocess the data using preprocess_data.py.
   This file extracts faces from each video from the dataset. Takes maximum 30 frames from each video (No. of frames depends on the length of the video.)
3. Building model: Next we will build a model. We will use Xception model, best for detecting subtle atrifacts in deepfake images.
4. Training the model: Time to train the model. This step generates a file with an extension (.h5). This is the model we needed to made the decisions.
5. Evaluation step: This step involves evaulation of the model. This checks the accuracy of the model.
6. Navigate to the webapp directory and run #python app.py
7. The webapp will be available at local host (127.0.0.1:5000)
8. Upload the video and click on analyse video
9. Result will be displayed in 2 mins.
