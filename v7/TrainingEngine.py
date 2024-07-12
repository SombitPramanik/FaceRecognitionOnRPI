import os
import face_recognition
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_face_recognition_model_from_folder(image_folder, model_output_path):
    all_face_encodings = []
    all_labels = []

    # Traverse through each subdirectory (each person)
    for person_name in os.listdir(image_folder):
        person_folder = os.path.join(image_folder, person_name)
        
        if not os.path.isdir(person_folder):
            continue

        # Process each image in the person's folder
        for image_file in os.listdir(person_folder):
            if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(person_folder, image_file)
                
                # Load the image
                image = face_recognition.load_image_file(image_path)

                # Detect faces in the image
                face_locations = face_recognition.face_locations(image)
                
                # Extract face encodings
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                # Append encodings and label (person's name)
                all_face_encodings.extend(face_encodings)
                all_labels.extend([person_name] * len(face_encodings))

    if not all_face_encodings:
        raise ValueError("No face encodings found in the provided images.")

    # Convert the face encodings and labels to numpy arrays
    X = np.array(all_face_encodings)
    y = np.array(all_labels)

    # Train a k-NN classifier
    knn_clf = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', weights='distance')
    knn_clf.fit(X, y)

    # Save the trained model
    joblib.dump(knn_clf, model_output_path)

    print(f"Model trained and saved to {model_output_path}")


filename = input("Enter the File Name : ")
# Example usage
train_face_recognition_model_from_folder("DataBase", filename)
