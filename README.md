This project outlines the development of a reliable face recognition-based attendance system that incorporates the face_recognition library, OpenCV and a Flask-based web application.  
The system is designed to detect and identify faces from live video feeds, maintain an attendance record and offer web-based management features.

# Methodology
For this project, we adopted a quantitative research design, focusing on the effectiveness of a face recognition-based attendance system.  
The primary goal was to analyze how accurately the system could recognize faces and log attendance automatically. 
Data was collected through real-time image capture from a computer camera, with the system processing and analyzing facial data using the face_recognition library.  We used different angles of photos from individuals to ensure the system could handle a variety of real-world conditions, such as different facial orientations.
 Figure:

![image](https://github.com/user-attachments/assets/1d387942-9783-4677-af89-1ba17a653763)

## Data collection methods
Registered Users Photos: A dataset of people’s photos, taken from multiple angles, was used to train the system.  
These images were stored in an "Images" directory to allow the system to recognize faces from different perspectives and under varying conditions.

## Dynamic Image Capture: 
During testing, the system captured live video footage to detect faces in real-time.  
The captured images were compared to the pre-loaded dataset to mark attendance.  
If the person was recognized, their attendance was logged.

## Implementation Details
Face Detection and Recognition: The face_recognition library was employed for detecting and encoding faces.  
Known face encodings (pre-registered images) were loaded from the "Images" directory, while any unknown faces detected during real-time video capture were saved dynamically to an "unknownFaces" directory for future recognition.

## Dynamic Registration: 
New faces detected in the video stream were added to the dataset through a manual registration process.  
A Flask route allowed users to register new individuals, linking their names to their corresponding face images, thus updating the dataset with new entries.

## Data Analysis Techniques
Attendance Logging: Once a face was recognized, the system automatically logged attendance by recording the person’s name, the time of attendance.  
The attendance logs were stored in Excel sheets using excel commands and were also optionally saved in SQLite database for easy data retrieval and future analysis.

## Web Application
Flask Interface: The system utilized Flask to provide a web interface for managing the attendance system (as shown in Figure). 

