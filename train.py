
import cv2
import os
import numpy as np
import json

# STEP 1: Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
student_names = {}

dataset_path = "dataset"

# STEP 2: Read student folders
for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    student_id, student_name = folder.split("_", 1)

    student_id = int(student_id)

    student_names[student_id] = student_name

    # STEP 3: Read student images
    for image_name in os.listdir(folder_path):

        image_path = os.path.join(
            folder_path, image_name
        )

        image = cv2.imread(
            image_path, cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        image = cv2.resize(image, (200, 200))

        faces.append(image)
        labels.append(student_id)

# STEP 4: Check images
if len(faces) == 0:
    print("No face images found!")
    exit()

# STEP 5: Train the model
recognizer.train(
    faces,
    np.array(labels, dtype=np.int32)
)

# STEP 6: Save trained model
recognizer.save("trainer.yml")

# STEP 7: Save student details
with open("students.json", "w") as file:
    json.dump(student_names, file)

print("--------------------------------")
print("Face Training Completed!")
print("Total Images:", len(faces))
print("Total Students:", len(student_names))
print("--------------------------------")
