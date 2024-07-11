import face_recognition
import pickle
import cv2
from flask import Flask, render_template, Response
import time
from joblib import Parallel, delayed

app = Flask(__name__)


def Recognizer(frame, data):
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    return boxes, names, frame


def VideoFeedGenerator():
    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open("encodings.pickle", "rb").read())

    vCap = cv2.VideoCapture(0)
    vCap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vCap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    start_time = time.time()
    num_frames = 0

    while True:
        success, frame = vCap.read()
        if not success:
            break

        num_frames += 1

        if num_frames % 2 != 0:
            frames = [frame]
            results = Parallel(n_jobs=10)(
                delayed(Recognizer)(frame, data) for frame in frames
            )

            boxes, names, OutputFrame = results[0]

            elapsed_time = time.time() - start_time
            fps = num_frames / elapsed_time

            for (top, right, bottom, left), name in zip(boxes, names):
                cv2.rectangle(OutputFrame, (left, top), (right, bottom), (0, 255, 225), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(
                    OutputFrame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                )

            cv2.putText(
                OutputFrame,
                f"FPS: {round(fps, 2)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            _, buffer = cv2.imencode(".png", OutputFrame)
            FinalFrame = buffer.tobytes()

            yield (
                b"--frame\r\n" b"Content-Type: image/png\r\n\r\n" + FinalFrame + b"\r\n"
            )

    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/VideoFeed")
def VideoFeed():
    return Response(
        VideoFeedGenerator(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def main():
    app.run(debug=True,port="8080")


if __name__ == "__main__":
    main()
