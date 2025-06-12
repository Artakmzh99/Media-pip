import cv2
import mediapipe as mp
import numpy as np
import time

class LandmarkDetector:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face mesh and pose detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,  # Increased to detect multiple faces
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
            
    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process face mesh
        face_results = self.face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw face mesh landmarks
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        thickness=1, color=(0, 255, 0)
                    )
                )
                
                # Draw face mesh contours
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        thickness=1, color=(0, 0, 255)
                    )
                )
        
        # Process pose
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=pose_results.pose_landmarks,
                connections=self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    thickness=2, circle_radius=4, color=(255, 0, 0)
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    thickness=2, color=(0, 255, 0)
                )
            )
        
        return frame
    
    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip the frame horizontally for a later selfie-view display
                frame = cv2.flip(frame, 1)
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display FPS
                cv2.putText(
                    processed_frame,
                    f"Press 'q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Display the frame
                cv2.imshow('Face and Body Landmarks', processed_frame)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = LandmarkDetector()
        detector.run()
    except Exception as e:
        print(f"An error occurred: {e}") 