import cv2
import streamlit as st
import pandas as pd
import datetime
import os
import numpy as np
from tensorflow.keras.models import load_model
from Database.db import Base , Employee, FaceMeta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from deepface import DeepFace


class_names = {}

# Connect to the database
engine = create_engine('postgresql://postgres:@localhost:5430/faceio')
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

# Query the database to get the number of classes, class names, and class values
employees = session.query(Employee).order_by(Employee.empl_no).all()
class_names = {}
for i, employee in enumerate(employees):
    class_names[i] = employee.full_name
    print(f"Class {i}: {class_names[i]}")

# Load the pre-saved model
model = load_model('transfer_learning_trained13_classes_face_cnn_model_data_science.h5')

# Create an attendance dataframe
filename = f'attendance/attendance_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv'

try:
    attendance_df = pd.read_csv(filename)
except FileNotFoundError:
    print("File does not exist, creating a new one")
    attendance_df = pd.DataFrame(columns=['Name', 'Time','Employee No.'])
except pd.errors.EmptyDataError:
    print("File is empty")
    attendance_df = pd.DataFrame(columns=['Name', 'Time','Employee No.']) # Replace with your column names
except pd.errors.ParserError:
    print("Unable to parse file")
    attendance_df = pd.DataFrame(columns=['Name', 'Time','Employee No.'])

# Define a function to update the attendance dataframe
def update_attendance(name,empl_no):
    global attendance_df
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    attendance_df = attendance_df.append({'Name': name, 'Time': time,'Employee No.' : empl_no}, ignore_index=True)
    attendance_df.to_csv(f'attendance/attendance_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv', index=False)
   
# Define a function to predict the class from a single frame
def predict_from_frame(frame, threshold=0.95, max_attempts=1):
    global attendance_df
    # Load the face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    def preprocess_image(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Crop face and resize to (160, 160)
        if len(face) > 0:
            (x, y, w, h) = face[0]
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)
            return face
        else:
            return None
    deepface = DeepFace.build_model("Facenet")
    face = preprocess_image(frame)
    if face is not None:
        # Get embedding for face
        embedding = deepface.predict(face)[0]

        # Query the database for face embeddings
        face_meta = session.query(FaceMeta).all()

        # Find closest match in database
        min_distance = float('inf')
        match_employee_no = None

        for face in face_meta:
            employee_embeddings = np.frombuffer(face.embedding, dtype=np.float32).reshape((-1, 128))

            # Calculate distances between target embedding and employee embeddings
            distances = np.linalg.norm(employee_embeddings - embedding, axis=1)

            # Find closest match
            closest_distance = np.min(distances)
            if closest_distance < min_distance:
                min_distance = closest_distance
                match_employee_no = face.empl_no
            
    
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
        
        # Do multiple predictions
        attempts = 0
        predicted_class_idx = None
        while attempts < max_attempts and predicted_class_idx is None:
            prediction = model.predict(face_roi)[0]
            predicted_class_idx = np.argmax(prediction)
            predicted_class_name = class_names.get(predicted_class_idx, 'unknown')
            predicted_class_prob = prediction[predicted_class_idx]
            if predicted_class_prob < threshold:
                predicted_class_idx = None
            attempts += 1
        
        # Draw the bounding box and predicted class name on the frame
        if predicted_class_idx is not None:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Update the attendance dataframe
            employee = session.query(Employee).filter_by(full_name=predicted_class_name).first()
            if employee:
                employee_id = employee.empl_no
                print(employee_id)
                if int(employee_id) == int(match_employee_no):
                    if not attendance_df['Name'].str.contains(predicted_class_name).any():
                        update_attendance(predicted_class_name,int(match_employee_no))
            else:
                print(f"No employee found with name {predicted_class_name}")
        else:
            # If all attempts fail, label as unknown
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, 'unknown', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
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


# Query the database to get the cohort names
cohorts = session.query(Employee.cohort).distinct().all()
cohort_names = [c[0] for c in cohorts]

# Allow the user to select a cohort
selected_cohort = st.selectbox('Select a cohort', cohort_names)
date = st.date_input('Select date')

# Generate the filename based on the selected cohort and date
filename = f'attendance/attendance_{date.strftime("%Y-%m-%d")}.csv'

# Check if the file exists
if os.path.exists(filename):
    # Load the attendance dataframe from the CSV file
    attendance_df = pd.read_csv(filename)
    
    # Add a new column to the dataframe to indicate the status of each employee
    attendance_df['status'] = attendance_df['Time'].apply(lambda x: 'Late' if x > '08:00:00' else 'On time')
    
    # Highlight the rows where the status is 'Late'
    def highlight_late(row):
        if row['status'] == 'Late':
            return ['background-color: yellow']*len(row)
        else:
            return ['']*len(row)

    st.write(f"Attendance for {selected_cohort} cohort on {date.strftime('%Y-%m-%d')}")
    st.write(attendance_df.style.apply(highlight_late, axis=1))
    if st.button(f"Submit data for {selected_cohort}"):
        scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
                    "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

        credentials = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
        client = gspread.authorize(credentials)

        spreadsheet = client.open('csv_to_sheet')
        worksheet = spreadsheet.sheet1  # assuming the data goes into the first sheet

        with open(f'attendance/attendance_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv', 'r') as file_obj:
            content = file_obj.read()
            rows = content.split('\n')
            data = [row.split(',') for row in rows if row]  # remove empty rows and split the values by comma
            worksheet.append_rows(data)
            st.success("Data Uploaded successfully!")
else:
    st.write(f"No attendance data found for {selected_cohort} cohort on {date.strftime('%Y-%m-%d')}")

    