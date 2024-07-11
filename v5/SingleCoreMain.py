from flask import Flask, render_template, Response
import os
import face_recognition as fr
import time
import cv2
import pickle

app = Flask(__name__)

camera = None
fps = 0
start_time = time.time()
frame_count = 0


def get_camera():
    global camera
    if camera is None:
        index = 0
        while camera is None and index < 5:  # Try up to 5 indices
            try:
                camera = cv2.VideoCapture(index)
                if not camera.isOpened():
                    camera.release()
                    camera = None
                else:
                    # For Laptops those support 720*1080 quality
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    # For Raspberry Pi
                    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    break
            except Exception as e:
                print(f"Failed to open camera at index {index}: {str(e)}")
                index += 1
                continue
    return camera


def FaceRecognition(frame, face_locations):
    # Make the output frame :
    output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Load the trained model
    with open("trained_model.pkl", "rb") as model_file:
        knn_clf = pickle.load(model_file)

    # Extract face encodings from the frame
    face_encodings = fr.face_encodings(output_frame, face_locations)

    # List to hold names with accuracy
    names_with_accuracy = []

    for encoding in face_encodings:
        # Predict the closest face in the trained model
        distances, indices = knn_clf.kneighbors([encoding], n_neighbors=1)
        name = knn_clf.classes_[indices[0][0]]
        accuracy = 100 * (1 - distances[0][0])

        # Only consider predictions with at least 50% accuracy
        if accuracy >= 40:
            names_with_accuracy.append(f"{name} : {round(accuracy, 2)}%")
        else:
            names_with_accuracy.append("Unknown")

    # If no face encodings were found, return "Unknown"
    if not names_with_accuracy:
        return ["Unknown"] * len(face_locations), output_frame

    # Return the names with accuracy as a list of strings
    return names_with_accuracy, output_frame


def VideoFeedGenerator():
    global fps, start_time, frame_count
    camera = get_camera()
    if camera is None:
        print("Camera not available.")
        return

    while True:
        s, frame = camera.read()
        if not s:
            break
        else:
            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:  # Update FPS every 10 frames
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                start_time = time.time()
                frame_count = 0

            # Draw FPS on the frame
            cv2.putText(
                frame,
                f"FPS: {round(fps, 2)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            # To turn on grayscale uncomment this below line
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect Face Locations in the frame
            face_locations = fr.face_locations(frame)

            # Recognize Face
            names_with_accuracy, output_frame = FaceRecognition(frame, face_locations)

            # Add the Rectangle on the Face
            for (top, right, bottom, left), name in zip(
                face_locations, names_with_accuracy
            ):

                if name == "Unknown":
                    cv2.rectangle(
                        output_frame, (left, top), (right, bottom), (0,0,255), 2
                    )
                    y = top - 15 if top - 15 > 15 else top + 15
                    cv2.putText(
                        output_frame,
                        name,
                        (left, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.rectangle(
                        output_frame, (left, top), (right, bottom), (255, 255, 0), 2
                    )
                    y = top - 15 if top - 15 > 15 else top + 15
                    cv2.putText(
                        output_frame,
                        name,
                        (left, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2,
                    )

            # Encode frame to PNG format
            ret, buffer = cv2.imencode(".png", output_frame)
            frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n" b"Content-Type: image/png\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def Index():
    return render_template("index.html")


@app.route("/Video")
def Video():
    return Response(
        VideoFeedGenerator(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stop", methods=["POST"])
def stop():
    print("Code successfully Terminated")
    global camera
    camera.release()
    os.kill(
        os.getpid(), 2
    )  # Command to terminate the total Process Basically a dam Kill Switch
    return "Code Successfully Terminated", 200  # just in case this is here because
    # if I remove it, it sometimes throws an error from Flask that
    # this block of code does not return anything


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
