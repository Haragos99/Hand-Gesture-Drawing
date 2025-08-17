from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image




base_options = python.BaseOptions(model_asset_path='models\\hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options = base_options,num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

def draw_image(image):
    detection_result = detector.detect(image)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    return annotated_image



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def draw_scribble_with_your_finge(frame,detection_result,x,y):
    try:
      # Get Data
      hand_landmarks_list = detection_result.hand_landmarks
      s = frame
      # Code to count numbers of fingers raised will go here
      numRaised = 0
      # for each hand...
      for idx in range(len(hand_landmarks_list)):
         # hand landmarks is a list of landmarks where each entry in the list has an x, y, and z in normalized image coordinates
         hand_landmarks = hand_landmarks_list[idx]
         # for each fingertip... (hand_landmarks 4, 8, 12, and 16)
         for i in range(8,21,4):
            # make sure finger is higher in image the 3 proceeding values (2 finger segments and knuckle)
            tip_y = hand_landmarks[i].y
            dip_y = hand_landmarks[i-1].y
            pip_y = hand_landmarks[i-2].y
            mcp_y = hand_landmarks[i-3].y
            if tip_y < min(dip_y,pip_y,mcp_y):
               numRaised += 1
         # for the thumb
         # use direction vector from wrist to base of thumb to determine "raised"

         annotated = np.copy(s)
         height, width, _ = annotated.shape

         tip_x = hand_landmarks[4].x
         dip_x = hand_landmarks[3].x
         pip_x = hand_landmarks[2].x
         mcp_x = hand_landmarks[1].x
         palm_x = hand_landmarks[0].x
         if mcp_x > palm_x:
            if tip_x > max(dip_x,pip_x,mcp_x):
               numRaised += 1
         else:
            if tip_x < min(dip_x,pip_x,mcp_x):
               numRaised += 1

      # Code to display the number of fingers raised will go here
      annotated_image = np.copy(frame)
      height, width, _ = annotated_image.shape
      text_x = int(hand_landmarks[8].x * width) 
      text_y = int(hand_landmarks[8].y * height) 
      # cv2.putText(img = annotated_image, text = str(numRaised) + " Fingers Raised",
      #     org = (text_x, text_y), fontFace = cv2.FONT_HERSHEY_DUPLEX,
      #     fontScale = 1, color = (0,0,255), thickness = 2, lineType = cv2.LINE_4)
      if x is not None and y is not  None:
        cv2.line(annotated_image,(text_x, text_y),(x,y),(0,0,255),thickness = 4)
      x = text_x
      y = text_y
      return annotated_image, x , y 
    except:
       return frame, x, y 


    
# Initialize the video capture object
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_count = 0
panel= np.full((480,640,3), 255, dtype=np.uint8)
x = None
start_time = time.time()  # Start measuring time
y = None
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    height, width, channels = frame.shape
    if not ret:
        print("Error: Failed to capture image.")
        break
    frame_count += 1
    bbox_array = np.zeros([height,width,4], dtype=np.uint8)    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    annotated_frame = draw_image(mp_image)
    detection_result = detector.detect(mp_image)
    panel,x , y = draw_scribble_with_your_finge(panel, detection_result,x , y)

    panel_ = cv2.cvtColor(panel, cv2.COLOR_RGB2RGBA)


    bbox_array[:,:,0:3] = annotated_frame
    alpha = 0.5
    dst = cv2.addWeighted(bbox_array, alpha , panel_, 1-alpha, 0) 
    #bbox_array [:,:,3]= panel
    cv2.imshow('Camera Feed', dst)
    if time.time() - start_time < 5:
       print(f"Speed of render: {frame_count} frame")

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()