from openai import OpenAI
import json

with open("config.json", "r") as config_file:
    config = json.load(config_file)
client = OpenAI(api_key=config["openai"]["api_key"])
    
def get_prompts():
    """
    Get prompts from prompts.json file.
    """
    with open("prompts.json", "r") as prompts_file:
        prompts = json.load(prompts_file)
        image_prompt = prompts["img_prompt"]
        secondary_prompt = prompts["secondary_prompt"]
        additional_questions = prompts["additional_questions"]
    return image_prompt, secondary_prompt, additional_questions

def process_image(llm_model, img_prompt, user_msg, img_url):
    """
    Process an image using the given model and prompt.
    """
    completion = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": img_prompt},
            {"role": "user", "content": [
                {"type": "image_url", 
                 "image_url": {
                    "url": img_url
                    },
                },
            ]},
            {"role": "user", "content":user_msg}
        ],
        temperature=0.2
    )
    return completion.choices[0].message.content

def ask_additional_questions(llm_model, img_url, output, new_prompt, questions):
    """
    Ask additional questions based on the output from the previous step.
    """
    prompt = f"{output} {new_prompt} {questions}"
    completion = client.chat.completions.create(
        model=llm_model,
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
    IMAGE_URL = "https://drive.google.com/uc?export=download&id=17yL8_Ml4KShHQOJv8MlzHCmeWVcXMYoY"
    USER_MESSAGE = "This is a Philips Ultrasound Machine, it's a CX50. There are scratches on the lens, which are affecting the quality of the image. The machine is currently off."
    MODEL = "gpt-4o"

    initial_prompt, secondary_prompt, additional_questions = get_prompts()
    init_output = process_image(MODEL, initial_prompt, USER_MESSAGE, IMAGE_URL)
    final_output = ask_additional_questions(MODEL, IMAGE_URL, init_output, secondary_prompt, additional_questions)
    print(final_output)

if __name__ == '__main__':
    main()



