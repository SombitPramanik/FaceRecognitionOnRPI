import os
import cv2
import face_recognition as fr
import time
from flask import Flask, render_template, Response
import threading
import queue
import multiprocessing as mp
import pickle
import joblib

app = Flask(__name__)
Camera = None
FPS = 0
StartTime = time.time()
FrameCount = 0

frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

vraFrame = mp.Manager().dict()  # Shared dictionary for frames


def GetCamera():
    global Camera
    if Camera is None:
        Index = 0
        while Camera is None and Index < 5:  # Try up to 5 indices
            try:
                Camera = cv2.VideoCapture(Index)
                if not Camera.isOpened():
                    Camera.release()
                    Camera = None
                else:
                    Camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    Camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    break
            except Exception as e:
                print(f"Failed to open camera at index {Index}: {str(e)}")
                Index += 1
                continue
    return Camera


def detect_faces(frame_queue, result_queue):
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break
            face_locations = fr.face_locations(frame)
            if not result_queue.full():
                result_queue.put(face_locations)
        except queue.Empty:
            continue


def LiveCamera(vraFrame):
    global FPS, StartTime, FrameCount
    Camera = GetCamera()
    if Camera is None:
        print("Camera not available.")
        return

    while True:
        s, Frame = Camera.read()
        if not s:
            break
        else:
            # Calculate FPS
            FrameCount += 1
            if FrameCount % 5 == 0:  # Update FPS every 5 frames
                ElapsedTime = time.time() - StartTime
                FPS = FrameCount / ElapsedTime
                StartTime = time.time()
                FrameCount = 0

            # Draw FPS on the frame
            cv2.putText(
                Frame,
                f"FPS: {round(FPS, 2)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # To turn on grayscale uncomment this below line
            Frame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)

            # Update global frame variable
            vraFrame["frame"] = Frame

            # Encode frame to PNG format
            ret, buffer = cv2.imencode(".png", Frame)
            frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n" b"Content-Type: image/png\r\n\r\n" + frame_bytes + b"\r\n"
        )


def FaceRecognition(vraFrame):
    detection_thread = threading.Thread(
        target=detect_faces, args=(frame_queue, result_queue)
    )
    detection_thread.start()
    # Load the trained model'
    # This is the Joblib Model 
    knn_clf = joblib.load("trained_model.joblib")
    # This is the code for the pickle Model that is not recommend but this is made for backup only purpose
    # with open("trained_model.pkl", "rb") as model_file:
    #     knn_clf = pickle.load(model_file)

    while True:
        Frame = vraFrame.get("frame", None)
        if Frame is None:
            continue
        else:
            # Put the frame in the queue for face detection
            if not frame_queue.full():
                frame_queue.put(Frame)

            face_locations = []
            # List to hold names with accuracy
            names_with_accuracy = []
            try:
                # Check if face locations are available
                face_locations = result_queue.get(timeout=0.01)
            except queue.Empty:
                pass

            RGBFrame = cv2.cvtColor(Frame, cv2.COLOR_GRAY2RGB)

            face_encodings = fr.face_encodings(RGBFrame, face_locations)

            for encoding in face_encodings:
                # Predict the closest face in the trained model
                distances, indices = knn_clf.kneighbors([encoding], n_neighbors=1)
                name = knn_clf.classes_[indices[0][0]]
                accuracy = 100 * (1 - distances[0][0])

                # Only consider predictions with at least 50% accuracy
                if accuracy >= 50:
                    names_with_accuracy.append(f"{name} : {round(accuracy, 2)}%")
                else:
                    names_with_accuracy.append("Unknown")

            # Draw rectangles around the Face locations

            # Add the Rectangle on the Face
            for (top, right, bottom, left), name in zip(
                face_locations, names_with_accuracy
            ):

                if name == "Unknown":
                    cv2.rectangle(Frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    y = top - 15 if top - 15 > 15 else top + 15
                    cv2.putText(
                        Frame,
                        name,
                        (left, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.rectangle(Frame, (left, top), (right, bottom), (255, 255, 0), 2)
                    y = top - 15 if top - 15 > 15 else top + 15
                    cv2.putText(
                        Frame,
                        name,
                        (left, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2,
                    )

            # Encode frame to PNG format
            ret, buffer = cv2.imencode(".png", Frame)
            frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n" b"Content-Type: image/png\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def Index():
    return render_template("index.html")


@app.route("/DroneEye")
def DroneEye():
    return Response(
        LiveCamera(vraFrame), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/Recognition")
def Recognition():
    return Response(
        FaceRecognition(vraFrame), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/Stop")
def Stop():
    try:
        global Camera
        Camera.release()
        print("Code Successfully Terminated")
    except Exception as camReeseE:
        print(f"Camera Release Error: {camReeseE}")
    os.kill(os.getpid(), 2)
    return "Code Terminated Successfully"


def main():
    print("Started Web server")
    app.run(debug=True, host="0.0.0.0", port="8080")
    return None


if __name__ == "__main__":
    live_camera_process = mp.Process(target=LiveCamera, args=(vraFrame,))
    face_recognition_process = mp.Process(target=FaceRecognition, args=(vraFrame,))

    live_camera_process.start()
    face_recognition_process.start()

    live_camera_process.join()
    face_recognition_process.join()

    main()
