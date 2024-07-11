import face_recognition
import imutils
import pickle
import time
import cv2
from joblib import Parallel, delayed

currentname = "unknown"
encodingsP = "TrainedModel.pkl"

print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())

vs = cv2.VideoCapture(0)
time.sleep(2.0)
start_time = time.time()
num_frames = 0


def recognize_faces(frame, data):
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["Encodings"], encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["Names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)

    return boxes, names


while True:
    _, frame = vs.read()
    frame = imutils.resize(frame, width=500)

    frames = [frame]
    results = Parallel(n_jobs=4)(
        delayed(recognize_faces)(frame, data) for frame in frames
    )

    boxes, names = results[0]

    num_frames += 1
    elapsed_time = time.time() - start_time
    fps = num_frames / elapsed_time

    for (top, right, bottom, left), name in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(
            frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

    cv2.putText(
        frame,
        f"Press 'q' to quit FPS: {round(fps, 2)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # quit when 'q' key is pressed
    if key == ord("q"):
        break
    

cv2.destroyAllWindows()
vs.release()
