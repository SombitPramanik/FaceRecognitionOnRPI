import os
import face_recognition
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_face_recognition_model_from_folder(image_folder, model_output_path):
    # List all images in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    all_face_encodings = []
    all_labels = []

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        # Load the image
        image = face_recognition.load_image_file(image_path)

        # Detect faces in the image
        face_locations = face_recognition.face_locations(image)
        
        # Extract face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Use the filename (without extension) as the label
        label = os.path.splitext(image_file)[0]

        # Append encodings and label
        all_face_encodings.extend(face_encodings)
        all_labels.extend([label] * len(face_encodings))

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

# Example usage
train_face_recognition_model_from_folder("Images", "trained_model.joblib")
