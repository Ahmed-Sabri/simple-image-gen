#pip install accelerate diffusers transformers torch
import requests
import random
import subprocess
import sys
from diffusers import StableDiffusionPipeline
import torch
import os

def install_dependencies():
    dependencies = ["accelerate", "diffusers", "transformers", "torch", "requests"]
    for package in dependencies:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure dependencies are installed
install_dependencies()

def get_available_models():
    url = "https://image.pollinations.ai/models"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch models from Pollinations AI.")
        return []

def download_image(image_url, model, image_num):
    response = requests.get(image_url)
    filename = f'{model}-image-{image_num}.jpg'
    with open(filename, 'wb') as file:
        file.write(response.content)
    print(f'Download Completed: {filename}')

def generate_pollinations_images(prompt, num_variations):
    models = get_available_models()
    if not models:
        print("No models found. Exiting Pollinations AI generation.")
        return
    
    for model in models:
        for i in range(1, num_variations + 1):
            seed = random.randint(1, 100000)
            image_url = f"https://pollinations.ai/p/{prompt}?width=768&height=768&seed={seed}&model={model}"
            download_image(image_url, model, i)

def generate_stable_diffusion_images(prompt, num_variations, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    for i in range(num_variations):
        print(f"Generating Stable Diffusion image for: {prompt}")
        image = pipe(prompt).images[0]
        output_path = os.path.join(output_directory, f"sd_image_{i + 1}.png")
        image.save(output_path)
        print(f"Image saved to {output_path}")

if __name__ == "__main__":
    prompt = input("Enter your prompt: ")
    num_variations = int(input("Enter the number of variations per model: "))
    choice = input("Would you like to use (1) Pollinations AI, (2) Stable Diffusion, or (3) Both? Enter 1, 2, or 3: ")
    
    if choice == "1":
        generate_pollinations_images(prompt, num_variations)
    elif choice == "2":
        generate_stable_diffusion_images(prompt, num_variations, "sd_images")
    elif choice == "3":
        generate_pollinations_images(prompt, num_variations)
        generate_stable_diffusion_images(prompt, num_variations, "sd_images")
    else:
        print("Invalid choice. Exiting.")
