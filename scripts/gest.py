from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math
import time



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


def image_with_gesture_and_hand_landmarks(image, result):
    image = image.numpy_view()
    gesture, multi_hand_landmarks = result

    # Size and spacing.
    FIGSIZE = 6.5  # Adjusted for a single image

    # Title with gesture and score

    title = f"{gesture.category_name} ({gesture.score:.2f})"
    annotated_image = image.copy()

    # Draw hand landmarks
    for hand_landmarks in multi_hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in hand_landmarks
        ])

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    cv2.putText(annotated_image, title ,
        (200, 200), cv2.FONT_HERSHEY_DUPLEX,
        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


base_options = python.BaseOptions(model_asset_path='models\\my_gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

input_video_path = "resorce\\M.mp4"
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video writer
output_video_path = "K.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


frame_count = 0
start_time = time.time()  # Start measuring time



# Process video frame by frame
while cap.isOpened():
    isreadable, frame = cap.read()
    if not isreadable:
        break
    frame_count += 1
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # STEP 4: Recognize gestures in the input image.
    recognition_result = recognizer.recognize(image)

    # STEP 5: Process the result. In this case, visualize it.
    if not recognition_result.gestures:
        out.write(frame_rgb)
        continue
    top_gesture = recognition_result.gestures[0][0]
    hand_landmarks = recognition_result.hand_landmarks
    results = ((top_gesture, hand_landmarks))
    annotated_frame = image_with_gesture_and_hand_landmarks(image, results)
    #out.write(annotated_frame)
    if frame_count >= fps:
      elapsed_time = time.time() - start_time  # Time taken to render 1 second of video
      print(f"Speed of render: {elapsed_time:.2f} sec")

      frame_count = 0  # Reset frame counter
      start_time = time.time()  # Reset timer


    cv2.imshow('Camera Feed', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()


