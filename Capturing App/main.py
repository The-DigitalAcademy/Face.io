import cv2
import os
import random
import streamlit as st
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from Database.db import Base, Employee
import psycopg2




# Define the capture function
def capture_photos(person_number):
    # Define the capture device
    cap = cv2.VideoCapture(0)
    # Set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Define the directory to save the images
    dir_path = 'photos/'
    # Get the person's unique number
    
    # Define the labels for the images
    labels = ['train', 'test', 'val']
    # Define the number of images to capture
    num_images = 30
    # Define the split ratio
    split_ratio = [0.7, 0.2, 0.1]
    # Define the path to the Haar Cascade Classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Loop through the labels
    # Loop through the labels
    for label in labels:
        # Create the directory for the label and person number
        label_dir_path = os.path.join(dir_path, label, f'{person_number}')
        os.makedirs(label_dir_path, exist_ok=True)
        # Loop through the number of images
        for i in range(num_images):
            # Read a frame from the capture device
            ret, frame = cap.read()
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the grayscale image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
            # Loop through the faces
            for (x, y, w, h) in faces:
                # Generate a random number between 0 and 1
                random_num = random.uniform(0, 1)
                # Determine the split based on the random number
                if random_num < split_ratio[0]:
                    split_label = 'train'
                elif random_num < split_ratio[0] + split_ratio[1]:
                    split_label = 'test'
                else:
                    split_label = 'val'
                # Extract the face from the frame
                face = frame[y:y+h, x:x+w]
                # Resize the face to 100x100 pixels
                face = cv2.resize(face, (100, 100))
                # Save the image to the appropriate directory
                img_path = os.path.join(label_dir_path, f'{person_number}_{split_label}_{i:02d}.jpg')
                cv2.imwrite(img_path, face)
            # Display the frame in the Streamlit app
            st.image(frame, channels='BGR', use_column_width=True)
    # Release the capture device
    cap.release()
    cv2.destroyAllWindows()

def detect_faces():
    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    # Start the webcam stream
    st.title("Webcam Live Feed")
    st.write("Please use this to find proper lighting")
    st.write("please untick run once you have good lighting and you see the green box around your face")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    # Loop until the "Run" checkbox is unchecked
    while run:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame using the Haar Cascade classifier
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the frame with the detected faces in the Streamlit app
        FRAME_WINDOW.image(frame)

    # Release the video capture object
    cap.release()
    st.write('Stopped')

def add_employee(emp_number, name, surname, cohort):
    engine = create_engine('postgresql://postgres:@localhost:5430/faceio')
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    new_employee = Employee(empl_no=emp_number, full_name=name + '_' + surname, cohort=cohort)
    session.add(new_employee)
    session.commit()
    st.success("Saved information successfully",icon="✅")
    return new_employee

# Create the Streamlit app
def app():
    st.title('Face Photo Capture')
    st.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    emp_number = st.text_input('Enter your Employee number:', '')
    name = st.text_input("Enter Name:")
    surname = st.text_input("Enter Surname:")
    cohort = st.selectbox(("Cohort"),
                              ("Data Science","Full Stack","Sales Force"))
    if st.button('Save Information'):
        add_employee(int(emp_number), name ,surname, cohort)
    st.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    st.text('Press the button below to start capturing photos')
    detect_faces()
    st.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    if st.button('Capture Photos'):
        capture_photos(person_number=emp_number)
        st.text('Photos captured successfully!')
    else:
        st.text('Click the button to start capturing photos')
    st.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
if __name__ == "__main__":
    app()