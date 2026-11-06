
import cv2
import csv
import os
import json
from datetime import datetime

# STEP 1: Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# STEP 2: Load student details
with open("students.json", "r") as file:
    student_names = json.load(file)

# STEP 3: Create attendance folder
os.makedirs("attendance", exist_ok=True)

today = datetime.now().strftime("%Y-%m-%d")

filename = f"attendance/attendance_{today}.csv"

# STEP 4: Create CSV file
if not os.path.exists(filename):

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Student ID",
            "Student Name",
            "Date",
            "Time",
            "Status"
        ])

# STEP 5: Read existing attendance
marked_ids = set()

with open(filename, "r", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:
        marked_ids.add(row["Student ID"])

# STEP 6: Load face detector
face_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(face_path)

if face_detector.empty():
    raise RuntimeError("Face detector XML not found!")

# STEP 7: Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Cannot access webcam!")

print("Attendance System Started")
print("Press Q to exit")

# STEP 8: Real-time face recognition
while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera reading failed!")
        break

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray, 1.3, 5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (200, 200))

        student_id, confidence = recognizer.predict(face)

        # Lower confidence distance means a better match
        if confidence < 70:

            name = student_names.get(
                str(student_id), "Unknown"
            )

            color = (0, 255, 0)

            if str(student_id) not in marked_ids:

                now = datetime.now()

                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                with open(filename, "a", newline="") as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        student_id,
                        name,
                        date,
                        time,
                        "Present"
                    ])

                marked_ids.add(str(student_id))

                print(name, "Attendance Marked!")

            label = name + " - Present"

        else:

            label = "Unknown"
            color = (0, 0, 255)

        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow(
        "Face Recognition Attendance System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# STEP 9: Close webcam
camera.release()
cv2.destroyAllWindows()

print("Attendance System Closed")
