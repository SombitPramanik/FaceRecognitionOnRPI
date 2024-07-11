import cv2

def capture_still_image(camera_index=0, output_filename="still_image.jpg"):
    # Open the camera
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Capture a frame
    ret, frame = cap.read()

    # Release the camera
    cap.release()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Could not capture frame.")
        return

    # Save the captured frame as a still image
    cv2.imwrite(output_filename, frame)
    print(f"Still image captured and saved as '{output_filename}'.")

# Call the function to capture a still image from camera at index 0
capture_still_image()
