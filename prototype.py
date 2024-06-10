from openai import OpenAI
import os
import json
import requests


MODEL = "gpt-4o"

with open("config.json", "r") as config_file:
    config = json.load(config_file)
client = OpenAI(api_key=config["openai"]["api_key"])
    
def get_prompts():
    with open("prompts.json", "r") as prompts_file:
        prompts = json.load(prompts_file)
        image_prompt = prompts["img_prompt"]
        secondary_prompt = prompts["secondary_prompt"]
        additional_questions = prompts["additional_questions"]
    return image_prompt, secondary_prompt, additional_questions

def process_image(img_prompt, img_url):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": img_prompt},
            {"role": "user", "content": [
                {"type": "image_url", 
                 "image_url": {
                    "url": img_url
                    },
                },
            ]},
        ],
        temperature=0.2
    )
    return completion.choices[0].message.content

def ask_additional_questions(img_url, output, new_prompt, questions):
    prompt = f"{output} {new_prompt} {questions}"
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "image_url", 
                 "image_url": {
                    "url": img_url
                    },
                },
            ]},
        ],
        temperature=0.2
    )
    return completion.choices[0].message.content

def main():
    IMAGE_URL = "https://drive.google.com/uc?export=download&id=1-0r7Gyfqu1xPbQ3fQHv5ROGXuyQJQRG8"
    initial_prompt, secondary_prompt, additional_questions = get_prompts()
    init_output = process_image(initial_prompt, IMAGE_URL)
    final_output = ask_additional_questions(IMAGE_URL, init_output, secondary_prompt, additional_questions)
    print("Assistant: " + init_output)
    print("Assistant: " + final_output)

if __name__ == '__main__':
    main()



