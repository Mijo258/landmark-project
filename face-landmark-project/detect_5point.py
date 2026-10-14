import os

import cv2
import dlib
import numpy as np

def main():
    # Resolve the model file relative to this script, so the program works
    # no matter which directory it is launched from.
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "shape_predictor_5_face_landmarks.dat")

    # Initialize Dlib's HOG-based face detector and the 5-point shape predictor
    print("[INFO] Loading facial landmark predictor...")
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(model_path)
    except Exception as e:
        print(f"[ERROR] Could not load model files. Ensure 'shape_predictor_5_face_landmarks.dat' is in the directory. Details: {e}")
        return

    # Initialize the laptop webcam (Index 0 is the default built-in camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not access the laptop camera.")
        return

    print("[INFO] Camera stream started. Press 'q' in the video window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from camera stream.")
            break

        # Dlib works on grayscale images for faster feature computation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = detector(gray, 0)

        for face in faces:
            # Extract bounding box coordinates for the face
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            # Draw a bounding box around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Predict the 5 facial landmarks (Right eye, Left eye, Nose tip, etc.)
            shape = predictor(gray, face)
            
            # Loop over the 5 landmark points and plot them
            for i in range(5):
                pt = shape.part(i)
                pt_coords = (pt.x, pt.y)
                
                # Draw a small red circle on each landmark
                cv2.circle(frame, pt_coords, 3, (0, 0, 255), -1)
                
                # Annotate the landmark index (0 to 4)
                cv2.putText(frame, str(i), (pt.x + 4, pt.y - 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Display the real-time feed
        cv2.imshow("5-Point Facial Landmark Detection", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed successfully.")

if __name__ == "__main__":
    main()