from openai import OpenAI
import os
import base64

os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

IMG_PATH = "sample_data/ultra.jpg"
image_base64 = encode_image(IMG_PATH)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Identify this image. what is it? See if you can find the make and model of this machine if it is a machine."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": 
             {"url": f"data:image/jpg;base64, {image_base64}"}
            }
        ]},
    ],
    temperature=0.0
)
print("Assistant: " + completion.choices[0].message.content)
