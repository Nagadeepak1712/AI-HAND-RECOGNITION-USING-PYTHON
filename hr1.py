import cv2
import mediapipe as mp
import numpy as np
import pygetwindow as gw
import ctypes

# Initialize Mediapipe components
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV
cap = cv2.VideoCapture(0)

# Constants for gesture recognition
DISTANCE_THRESHOLD = 0.1  # Adjust this threshold as needed
BRIGHTNESS_INCREMENT = 0.1  # Adjust this increment as needed

def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def is_hand_open(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    thumb_index_distance = calculate_distance(thumb_tip, index_tip)
    return thumb_index_distance > DISTANCE_THRESHOLD

def set_brightness(brightness):
    # Windows-specific brightness control (placeholder, replace with actual implementation)
    # For demonstration purposes only, use a tool like `screen-brightness-control` for real implementation
    brightness = max(0, min(1, brightness))  # Clamp brightness between 0 and 1
    print(f"Setting brightness to {brightness * 100:.2f}%")  # Placeholder action

def main():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Determine if the hand is open
                if is_hand_open(hand_landmarks):
                    # Here, we're just incrementing brightness for demonstration
                    current_brightness = 0.5  # Placeholder value
                    new_brightness = min(current_brightness + BRIGHTNESS_INCREMENT, 1.0)
                    set_brightness(new_brightness)
                else:
                    # Adjust brightness if needed
                    current_brightness = 0.5  # Placeholder value
                    new_brightness = max(current_brightness - BRIGHTNESS_INCREMENT, 0.0)
                    set_brightness(new_brightness)

        # Display the frame
        cv2.imshow('Hand Gesture Control', frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
