
import cv2
import os

# STEP 1: Get student details
student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")

# STEP 2: Create student folder
folder = f"dataset/{student_id}_{student_name}"
os.makedirs(folder, exist_ok=True)

# STEP 3: Load face detector
face_path = (
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(face_path)

if face_detector.empty():
    print("Error: Face detection XML not found!")
    exit()

# STEP 4: Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Cannot open webcam!")
    exit()

count = 0

print("Look at the camera. Press Q to stop.")

# STEP 5: Capture face images
while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera reading failed!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray, 1.3, 5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        if count < 50:
            count += 1

            cv2.imwrite(
                f"{folder}/{count}.jpg", face
            )

        cv2.rectangle(
            frame, (x, y), (x+w, y+h),
            (0, 255, 0), 2
        )

    cv2.putText(
        frame, f"Images: {count}/50",
        (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2
    )

    cv2.imshow("Student Registration", frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    if count >= 50:
        break

camera.release()
cv2.destroyAllWindows()

print("Registration completed!")
