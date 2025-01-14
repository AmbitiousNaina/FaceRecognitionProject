from flask import Blueprint, Response, render_template, request
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
from database.sqlite_operations import save_attendance

# Define the blueprint for video streaming
video_stream_bp = Blueprint('video_stream_bp', __name__)

# Initialize video capture and known faces list
video_capture = cv2.VideoCapture(0)
known_face_encodings = []
known_face_names = []
detected_faces = {}

detected: bool = 0

# Directory paths
Images_dir = os.path.join(os.getcwd(), 'Images')
new_faces_dir = os.path.join(os.getcwd(), 'new_faces')  # Directory for saving new faces

# Load known faces from 'Images' folder dynamically
def load_Images():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    # Make sure new_faces_dir exists
    if not os.path.exists(new_faces_dir):
        os.makedirs(new_faces_dir)

    for filename in os.listdir(Images_dir):
        if filename.endswith(".jpg") or filename.endswith(".avif"):
            image = face_recognition.load_image_file(os.path.join(Images_dir, filename))
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(filename.split('.')[0])  # Use filename (without extension) as name

load_Images()  # Load faces initially

# Function to generate video frames with face recognition
def gen_frames():
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        # Draw rectangles and labels on faces
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            current_time = datetime.now()

            if name == 'Unknown':
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                accuracy = int(100 - (min(face_distances) * 100))
                cv2.putText(frame, f"{name}, {accuracy}%", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                accuracy = int(100 - (min(face_distances) * 100))
                detected = 1
                cv2.putText(frame, f"{name}, {accuracy}%", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                # Log attendance for recognized faces with accuracy
                save_attendance(name, current_time.now(), accuracy)
                detected_faces[name] = current_time  # Store the timestamp for recognized faces

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route to handle video stream
@video_stream_bp.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to register new students (add face encodings)
@video_stream_bp.route('/register-student', methods=['GET', 'POST'])
def register_student():
    if request.method == 'POST':
        student_name = request.form['name']
        image_file = request.files['image']
        image_path = os.path.join(Images_dir, f"{student_name}.jpg")

        # Save image to 'Images' folder
        image_file.save(image_path)

        # Load and encode the new face
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]

        # Save face encoding and name for future recognition
        known_face_encodings.append(encoding)
        known_face_names.append(student_name)

        # Optionally, save attendance to the database
        save_attendance(student_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 100)  # 100% accuracy

        return render_template('register_new.html', message="Student registered successfully!")

    return render_template('register_new.html')

# Route to handle registration of unknown faces (through webcam feed)
@video_stream_bp.route('/register-unknown-face', methods=['GET', 'POST'])
def register_unknown_face():
    global known_face_encodings, known_face_names

    if request.method == 'POST':
        # Capture an unknown face image
        student_name = request.form['name']

        # Save the unknown face to a file
        image_path = os.path.join(new_faces_dir, f"{student_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        ret, frame = video_capture.read()
        cv2.imwrite(image_path, frame)

        # Load and encode the new face
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]

        # Add the new face encoding to the known faces list
        known_face_encodings.append(encoding)
        known_face_names.append(student_name)

        # Optionally, save attendance to the database
        save_attendance(student_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 100)  # 100% accuracy

        return render_template('register_new.html', message="Unknown face registered successfully!")

    return render_template('register_new.html')
