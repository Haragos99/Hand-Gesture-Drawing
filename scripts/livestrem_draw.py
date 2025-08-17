import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constants
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_COLOR = (88, 205, 54)  # vibrant green
LINE_COLOR = (0, 0, 255)
LINE_THICKNESS = 4
ALPHA_BLEND = 0.5

# Setup Mediapipe Detector
base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Drawing Helpers
def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand landmarks and handedness on an image."""
    annotated_image = np.copy(rgb_image)
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
        # Draw landmarks
        hand_proto = landmark_pb2.NormalizedLandmarkList()
        hand_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Add handedness text
        h, w, _ = annotated_image.shape
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coords) * w)
        text_y = int(min(y_coords) * h) - MARGIN
        cv2.putText(
            annotated_image,
            handedness[0].category_name,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA
        )

    return annotated_image


def draw_image(mp_image):
    """Run hand detection and annotate image."""
    detection_result = detector.detect(mp_image)
    return draw_landmarks_on_image(mp_image.numpy_view(), detection_result), detection_result

def is_pointing(hand_landmarks):
    """Return True if the index finger is pointing (straight and extended)."""
    tip_y = hand_landmarks[8].y
    dip_y = hand_landmarks[7].y
    pip_y = hand_landmarks[6].y
    mcp_y = hand_landmarks[5].y

    return tip_y < min(dip_y, pip_y, mcp_y)

# Finger Scribble
def draw_scribble_with_finger(panel, detection_result, prev_x, prev_y):
    """Draw scribble following the index finger."""
    try:
        annotated_panel = np.copy(panel)
        for hand_landmarks in detection_result.hand_landmarks:
            h, w, _ = annotated_panel.shape

            if is_pointing(hand_landmarks): 
              # Get index fingertip position
              index_finger_tip = hand_landmarks[8]
              x = int(index_finger_tip.x * w)
              y = int(index_finger_tip.y * h)

              # Draw line from previous position to current
              if prev_x is not None and prev_y is not None:
                  cv2.line(annotated_panel, (x, y), (prev_x, prev_y), LINE_COLOR, thickness=LINE_THICKNESS)
              return annotated_panel, x, y

        return panel, prev_x, prev_y
    except Exception:
        return panel, prev_x, prev_y

def livestream():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_count, start_time = 0, time.time()
    panel = np.full((480, 640, 3), 255, dtype=np.uint8)
    prev_x, prev_y = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run detection + drawing
        annotated_frame, detection_result = draw_image(mp_image)
        panel, prev_x, prev_y = draw_scribble_with_finger(panel, detection_result, prev_x, prev_y)

        # Blend annotated frame and scribble panel
        blended = cv2.addWeighted(annotated_frame, ALPHA_BLEND, panel, 1 - ALPHA_BLEND, 0)

        # Show results
        cv2.imshow('Hand Tracking Scribble', blended)
        frame_count += 1

        # Print FPS during first 5 seconds
        if time.time() - start_time < 5:
            print(f"Speed of render: {frame_count} frames")

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    livestream()
