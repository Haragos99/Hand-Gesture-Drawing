from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

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
