#from flask_ngrok import run_with_ngrok
from flask_ngrok2 import run_with_ngrok
from flask import Flask, render_template, request

import torch
from diffusers import StableDiffusionPipeline

import base64
from io import BytesIO

# Load model
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large",torch_dtype=torch.float16)
#pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v3-5", variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

# Start flask app and set to ngrok
app = Flask(__name__)
#run_with_ngrok(app)
run_with_ngrok(app=app, auth_token="30TDlr4bckXD5uo9p8YCDFQILEH_7ohXgPHUv9NTn5H8fEw3")

@app.route('/')
def initial():
  return render_template('index.html')


@app.route('/submit-prompt', methods=['POST'])
def generate_image():
  prompt = request.form['prompt-input']
  print(f"Generating an image of {prompt}")

  #image = pipe(prompt).images[0]
  image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

  print("Image generated! Converting image ...")
  
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue())
  img_str = "data:image/png;base64," + str(img_str)[2:-1]

  print("Sending image ...")
  return render_template('index.html', generated_image=img_str, given_prompt=prompt)


if __name__ == '__main__':
    app.run()