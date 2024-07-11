import os
import cv2
import face_recognition
import mysql.connector
import numpy as np

# MySQL database connection
def connect_to_db():
    return mysql.connector.connect(
        host="Mysql Server Address",
        user="mysql'sName",
        password="MysqlPassword",
        database="DataBaseName"
    )

# Function to insert face data into the database
def insert_face_data(name, encoding):
    db = connect_to_db()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO FaceData (Name, Encodings) VALUES (%s, %s)",
        (name, ','.join(str(x) for x in encoding))
    )
    db.commit()
    cursor.close()
    db.close()

# Function to process images in a folder
def process_images(image_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(image_folder, filename)
            name, _ = os.path.splitext(filename)
            
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Detect face and extract encoding
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                # Assume the first face detected is the correct one
                encoding = face_encodings[0]
                insert_face_data(name, encoding)
                print(f"Processed {filename} and added to database.")
            else:
                print(f"No faces found in {filename}.")

# Define the image folder
image_folder = 'Images'

# Process images and upload data to MySQL
process_images(image_folder)
