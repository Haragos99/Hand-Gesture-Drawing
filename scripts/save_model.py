from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
import torch
import transformers

# Define the directory where you want to save the models
model_save_directory = "C:\\UX\\model_directory"

# Define model IDs
base_model_id = "runwayml/stable-diffusion-v1-5"
controlnet_model_id = "lllyasviel/control_v11p_sd15_scribble"

# Load and save the ControlNet model
controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16)
controlnet.save_pretrained(f"{model_save_directory}/controlnet")

# Load and save the Stable Diffusion pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_id, controlnet=controlnet, torch_dtype=torch.float16
)
pipe.save_pretrained(f"{model_save_directory}/stable_diffusion_pipeline")
