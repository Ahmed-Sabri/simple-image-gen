#pip install accelerate diffusers transformers torch
from diffusers import StableDiffusionPipeline
import torch
import os

def generate_images_with_stable_diffusion(prompts, output_directory):
    """
    Generate images for given prompts using Stable Diffusion.
    
    Args:
        prompts (list): A list of textual descriptions for the images.
        output_directory (str): Directory to save the generated images.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Load Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    for i, prompt in enumerate(prompts):
        print(f"Generating image for prompt: {prompt}")
        image = pipe(prompt).images[0]
        output_path = os.path.join(output_directory, f"image_{i + 1}.png")
        image.save(output_path)
        print(f"Image saved to {output_path}")

if __name__ == "__main__":
    # Example prompts
    prompts = [
        "A fridge with its door slightly open",
        "An oven with steam coming out",
        "A washing machine with clothes inside",
        "A dishwasher with plates stacked",
        "A steam cleaner cleaning a 6x9 area rug",
        "A steam cleaner cleaning a 9x13 area rug",
        "A staircase being cleaned with a steam cleaner",
        "An organized cabinet with open doors",
        "A small additional kitchen with utensils and stove",
        "A bed with freshly folded sheets",
        "A steam cleaner on a bedroom carpet",
        "A paw print symbolizing a pet fee"
    ]
    
    # Output directory
    output_directory = "stable_diffusion_images"
    
    # Generate images
    generate_images_with_stable_diffusion(prompts, output_directory)
