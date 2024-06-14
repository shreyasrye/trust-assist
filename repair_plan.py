import json
import diagnostic
from openai import OpenAI

FLOWCHART_URL = "https://bmet.ewh.org/server/api/core/bitstreams/ac88d555-d4ba-4950-9250-1f4bfb9a9556/content"
QUESTION = "Based on this initial diagnosis, and the fact that this issue is not remotely solvable, create a repair plan, and order parts needed. please provide the repair plan based on this flowchart:"

with open("config.json", "r") as config_file:
    config = json.load(config_file)
client = OpenAI(api_key=config["openai"]["api_key"])

def generate_repair_plan(init_diagnosis, context, query):
    """
    Generates a repair plan based on the initial diagnosis and context (provided flowchart).
    """
    prompt = f"{init_diagnosis} \n {query} {context}"
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0.0
    )
    return completion.choices[0].message.content
