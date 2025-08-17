import cv2
import time
from pathlib import Path
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


#Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


#Constants
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (88, 205, 54)  # vibrant green


def annotate_hand_landmarks(image, gesture, hand_landmarks):
    """
    Draw landmarks and overlay gesture name + score on an image.
    """
    annotated_image = image.copy()

    # Convert hand landmarks to protobuf format
    for landmarks in hand_landmarks:
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in landmarks
        ])
        mp_drawing.draw_landmarks(
            annotated_image,
            landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    # Title with gesture and score
    title = f"{gesture.category_name} ({gesture.score:.2f})"
    cv2.putText(
        annotated_image,
        title,
        (200, 200),
        cv2.FONT_HERSHEY_DUPLEX,
        FONT_SIZE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )

    return annotated_image


def process_video(input_path: Path, output_path: Path, model_path: Path):
    """
    Process a video frame by frame using Mediapipe GestureRecognizer.
    Annotates the video with recognized user gestures and saves the result as a new video.
    """

    #Load mode
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    #Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #Define output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to Mediapipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Recognize gestures
        result = recognizer.recognize(mp_image)

        if not result.gestures:
            out.write(frame)
            continue

        top_gesture = result.gestures[0][0]
        annotated_frame = annotate_hand_landmarks(frame, top_gesture, result.hand_landmarks)

        out.write(annotated_frame)

        # Print processing speed once per second
        if frame_count >= fps:
            elapsed_time = time.time() - start_time
            print(f"Processing speed: {elapsed_time:.2f} sec per {fps} frames")
            frame_count = 0
            start_time = time.time()

        # Display preview
        cv2.imshow("Gesture Recognition", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    #Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    base_dir = Path("models")
    video_input = Path("resorce/M.mp4")
    video_output = Path("K.mp4")
    model_file = base_dir / "my_gesture_recognizer.task"

    process_video(video_input, video_output, model_file)
