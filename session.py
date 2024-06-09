from openai import OpenAI
import os
import base64
import json
import openai


IMG_PATH = "sample_data/ultra2.png"
MODEL = "gpt-4o"
session_store = {}

os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

with open("config.json", "r") as config_file:
    config = json.load(config_file)
    
def get_prompts():
    with open("prompts.json", "r") as prompts_file:
        prompts = json.load(prompts_file)
        image_prompt = prompts["img_prompt"]
        additional_questions = prompts["additional_questions"]
    return image_prompt, additional_questions

def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def process_image(img_prompt, encoded_image):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": img_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": 
                {"url": f"data:image/png;base64, {encode_image}"}
                }
            ]},
        ],
        temperature=0.0
    )
    return completion.choices[0].message.content

def save_session(session_id, data):
    session_store[session_id] = data

def get_session(session_id):
    return session_store.get(session_id)


initial_prompt, additional_questions = get_prompts()
image_base64 = encode_image(IMG_PATH)

print(process_image(initial_prompt, image_base64))

