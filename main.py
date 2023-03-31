import cv2
import streamlit as st
import pandas as pd
import datetime
import os
import numpy as np
from tensorflow.keras.models import load_model

num_classes = 4

# Load the pre-saved model
model = load_model('transfer_learning_trained5_classes_face_cnn_model.h5')

# Define a dictionary of class names
class_names = {0: 'chris_evans', 1: 'chris_hemsworth', 2: 'mark_ruffalo', 3: 'robert_downey_jr', 4: 'scarlett_johansson'}

# Create an attendance dataframe
try:
    attendance_df = pd.read_csv('attendance.csv')
except FileNotFoundError:
    attendance_df = pd.DataFrame(columns=['Name', 'Time'])

# Define a function to update the attendance dataframe
def update_attendance(name):
    global attendance_df
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    attendance_df = attendance_df.append({'Name': name, 'Time': time}, ignore_index=True)
    attendance_df.to_csv('attendance.csv', index=False)

# Define a function to predict the class from a single frame
def predict_from_frame(frame, threshold=0.9):
    global attendance_df
    # Load the face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    # Loop through the detected faces and do predictions
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (224, 224))
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.repeat(face_roi, 3, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)
        
        # Normalize pixel values
        face_roi = face_roi / 255.0
        
        # Do prediction
        prediction = model.predict(face_roi)[0]
        predicted_class = np.argmax(prediction)
        predicted_class_name = class_names.get(predicted_class, 'unknown')
        predicted_class_prob = prediction[predicted_class]
        
        # Check if the predicted class probability is above the threshold
        if predicted_class_prob >= threshold:
            # Draw the bounding box and predicted class name on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Update the attendance dataframe
            for x in range(num_classes):
    
                if attendance_df['Name'].str.contains(predicted_class_name).any():
                    pass
                else:
                    update_attendance(predicted_class_name)
            
        else:
            # If below threshold, draw unknown label on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    print(attendance_df)
    return frame


# Create the Streamlit app
st.title('Face Recognition')
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame = predict_from_frame(frame)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')

camera.release()
cv2.destroyAllWindows()
