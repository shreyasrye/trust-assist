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
        additional_questions = prompts["additional_questions"]
    return image_prompt, additional_questions

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

def ask_additional_questions(img_url, output, questions):
    new_prompt = f"{output} Based on this output, use these additional questions on the image to see if any more information can be gained on the issue (if there is an issue, otherwise disregard the questions. Also, by questions I mean the keys in the dictionary, and store the respective answers (the values of the dictionary) as the answer). Make sure to only use the questions/answers that are still needed, not those that we already have answers to: {questions}"
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            # do the same here but with the new prompt, the image and the additional questions
            {"role": "system", "content": new_prompt},
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
    initial_prompt, additional_questions = get_prompts()
    init_output = process_image(initial_prompt, IMAGE_URL)
    final_output = ask_additional_questions(IMAGE_URL, init_output, additional_questions)
    print("Assistant: " + final_output)

if __name__ == '__main__':
    main()



