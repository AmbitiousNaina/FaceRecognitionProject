import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime, date
from excel_commands import ExcelOperations

# Set up video capture from webcam
video_capture = cv2.VideoCapture(0)

# Directory where images of known people are stored
Images_dir = "Images"
new_faces_dir = "new_faces"  # Directory to save newly detected faces

# Initialize the lists to hold face encodings and names
known_face_encodings = []
known_face_names = []

# Load images from the Images directory and encode the faces
for image_filename in os.listdir(Images_dir):
    if image_filename.endswith((".jpg", ".jpeg")):
        image_path = os.path.join(Images_dir, image_filename)
        person_name = os.path.splitext(image_filename)[0]

        img = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(img)

        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(person_name)
        else:
            print(f"No face encodings found for {person_name} in {image_filename}")

# Check if we have any known faces loaded
if not known_face_encodings:
    print("No known faces found in the 'Images' directory.")
    exit()

# Initialize variables for video capture and face recognition
face_locations = []
face_encodings = []
face_names = []
detected = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    if process_this_frame:
        # Resize frame for faster processing (if needed)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Detect faces and get their locations and encodings
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        # Identify detected faces and compare with known faces
        face_names = []  # Reset face names list for each frame
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                # Find the best match if there are multiple faces matched
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                name = known_face_names[best_match_index]

            face_names.append(name)

            # If the face is unknown, prompt the user to add it
            if name == "Unknown":
                print("Unknown face detected!")
                # Ask the user for a name for the unknown face
                name_input = input("Please enter the name for this face: ")

                # Save the image of the unknown face to the new_faces directory
                if not os.path.exists(new_faces_dir):
                    os.makedirs(new_faces_dir)

                new_face_filename = os.path.join(new_faces_dir, f"{name_input}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(new_face_filename, frame)  # Save the current frame with the unknown face

                print(f"Saved new face as {new_face_filename}")

                # Add the new face encoding and name to the lists
                known_face_encodings.append(face_encoding)
                known_face_names.append(name_input)

                # Optionally, log the new face to an Excel sheet (or database)
                new_row = [name_input, datetime.now(), 100]  # You can adjust accuracy if needed
                koi = ExcelOperations()
                koi.create_or_append_excel(file_name=f"{date.today()}.xlsx", rows_to_add=new_row)
                print(f"Logged new face {name_input} to the Excel sheet.")

    process_this_frame = not process_this_frame

    # Draw rectangles and labels around detected faces
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw rectangles for each face
        if name == "Unknown":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Label each face with its name or "Unknown"
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Log to Excel for non-duplicate faces
        if name != "Unknown" and name not in detected:
            accuracy = int(100 - (min(face_distances) * 100))
            new_row = [name, datetime.now(), accuracy]
            koi = ExcelOperations()
            koi.create_or_append_excel(file_name=f"{date.today()}.xlsx", rows_to_add=new_row)
            print(f"{name} at {datetime.now()} with accuracy: {accuracy}")
            detected.append(name)

    # Show the video frame with rectangles and labels
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
