import face_attend
import cv2
import numpy as np
import sqlite3
from datetime import datetime

# Connect to SQLite database
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

# Create a table for attendance if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS Attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, date TEXT, time TEXT)''')

# Load known face encodings and their names
known_face_encodings = [...]  # Your known face encodings
known_face_names = [...]  # Your known face names

# Initialize some variables
face_locations = []
face_encodings = []
process_this_frame = True

# Capture video from webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_attend.face_locations(rgb_small_frame)
        face_encodings = face_attend.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_attend.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_attend.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                # Check if attendance is already marked for today
                today_date = datetime.now().strftime("%Y-%m-%d")
                cursor.execute("SELECT * FROM Attendance WHERE name=? AND date=?", (name, today_date))
                result = cursor.fetchall()

                if len(result) == 0:
                    # Mark attendance
                    now = datetime.now()
                    time_string = now.strftime("%H:%M:%S")
                    cursor.execute("INSERT INTO Attendance (name, date, time) VALUES (?, ?, ?)", (name, today_date, time_string))
                    conn.commit()
                    print(f"Attendance marked for {name} at {time_string}")

    process_this_frame = not process_this_frame

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit '?' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('?'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
conn.close()
