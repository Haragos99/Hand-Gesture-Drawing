

import random
import numpy as np
import cv2
import torch
import gradio as gr
import controlnet_aux
import PIL.Image
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

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




base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
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





MAX_NUM_IMAGES = 5
DEFAULT_NUM_IMAGES = 3
MAX_IMAGE_RESOLUTION = 768
DEFAULT_IMAGE_RESOLUTION = 768

MAX_SEED = np.iinfo(np.int32).max





# Define the local paths
base_model_path = "C:\\UX\\model_directory\\stable_diffusion_pipeline"
controlnet_model_path = "C:\\UX\\model_directory\\controlnet"

# Load the ControlNet model
controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)

# Load the Stable Diffusion pipeline with the ControlNet model
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16
)

# Move the pipeline to the desired device
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)



def resize_image(input_image, resolution, interpolation=None):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / max(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    # area interpolation for downsizing, lanczos for upsizing
    if interpolation is None:
        interpolation = cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    img = cv2.resize(input_image, (W, H), interpolation=interpolation)
    return img

 #image generation from scribble input
# using torch.inference_mode to disable gradient tracking
@torch.inference_mode()
def process_scribble_interactive(
    image_and_mask: dict[str, np.ndarray],
    prompt: str,
    additional_prompt: str,
    negative_prompt: str,
    num_images: int,
    image_resolution: int,
    num_steps: int,
    guidance_scale: float,
    seed: int,
) -> list[PIL.Image.Image]:
    if image_and_mask is None:
        raise ValueError
    if image_resolution > MAX_IMAGE_RESOLUTION:
        raise ValueError
    if num_images > MAX_NUM_IMAGES:
        raise ValueError

    image = image_and_mask["mask"]
    image = controlnet_aux.util.HWC3(image)
    image = resize_image(image, resolution=image_resolution)
    control_image = PIL.Image.fromarray(image)

    if not prompt:
        prompt = additional_prompt
    else:
        prompt = f"{prompt}, {additional_prompt}"

    generator = torch.Generator().manual_seed(seed)
    results = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        num_inference_steps=num_steps,
        generator=generator,
        image=control_image,
    ).images#
    return [control_image] + results




# random seed utility
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


# create Gradio-based user interface
# based on: https://huggingface.co/spaces/hysts/ControlNet
def create_demo(process):
    # UI components

    with gr.Blocks() as demo:
        white_image = np.full(
            shape=(DEFAULT_IMAGE_RESOLUTION, DEFAULT_IMAGE_RESOLUTION, 3),
            fill_value=255,
            dtype=np.uint8,
        )
        with gr.Row():
            gr.Markdown("# Image Generation Tool")
        with gr.Row():
            image = gr.Image(tool="sketch", brush_radius=10, label="Draw",value= white_image)
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
        with gr.Row():
            a_prompt = gr.Textbox(label="Additional Prompt")
            n_prompt = gr.Textbox(label="Negative Prompt")
        with gr.Row():
            num_samples = gr.Slider(label="Number of Images", minimum=1, maximum=MAX_NUM_IMAGES, value=DEFAULT_NUM_IMAGES, step=1)
            image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=MAX_IMAGE_RESOLUTION, value=DEFAULT_IMAGE_RESOLUTION, step=256)
        with gr.Row():
            num_steps = gr.Slider(label="Number of Steps", minimum=1, maximum=100, value=1, step=1)
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=1.0, step=0.1)
        with gr.Row():
            seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
        with gr.Row():
            run_button = gr.Button("Generate Image")
        with gr.Row():
            result = gr.Gallery(label="Output", columns=2, object_fit="scale-down")

               # UI behaviour
        inputs = [
            image,
            prompt,
            a_prompt,
            n_prompt,
            num_samples,
            image_resolution,
            num_steps,
            guidance_scale,
            seed,
        ]
        prompt.submit(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=process,
            inputs=inputs,
            outputs=result,
            api_name=False,
        )
        run_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=process,
            inputs=inputs,
            outputs=result,
        )
    return demo


demo = create_demo(process_scribble_interactive)
demo.queue().launch(debug=True)  # 40 s

# Hint: test the demo using the public URL given in the output of this cell