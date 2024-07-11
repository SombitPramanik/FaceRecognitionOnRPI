import atexit
import cv2
from flask import Flask, render_template, Response
import time

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
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    break
            except Exception as e:
                print(f"Failed to open camera at index {index}: {str(e)}")
                index += 1
                continue
    return camera

def generate():
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
            cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Encode frame to PNG format
            ret, buffer = cv2.imencode(".png", frame)
            frame_bytes = buffer.tobytes()
        
        yield (b"--frame\r\n" b"Content-Type: image/png\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def index():
    return render_template("stream.html")

@app.route("/video")
def video():
    return Response(generate(), mimetype="multipart/x-mixed-replace;boundary=frame")


def release_camera():
    camera.release()

    
if __name__ == "__main__":
    app.run(debug=True)
    # Release camera when the Flask app is terminated
    atexit.register(release_camera)