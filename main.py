import diagnostic
import remote_troubleshooting
import repair_plan
import json

def initial_diagnosis(image_url, user_msg):
    initial_prompt, secondary_prompt, additional_questions = diagnostic.get_prompts()
    return diagnostic.process_image("gpt-4o", initial_prompt, user_msg, image_url)

def secondary_diagnosis(image_url, init_diag):
    initial_prompt, secondary_prompt, additional_questions = diagnostic.get_prompts()
    return diagnostic.ask_additional_questions("gpt-4o", image_url, init_diag, secondary_prompt, additional_questions)

def create_diagnostic_report(input_query):
    return remote_troubleshooting.main(input_query)

def create_repair_plan(init_diagnosis, context, query):
    return repair_plan.generate_repair_plan(init_diagnosis, context, query)


IMAGE_URL = "https://drive.google.com/uc?export=download&id=17yL8_Ml4KShHQOJv8MlzHCmeWVcXMYoY"
USER_MESSAGE = "This is a Philips Ultrasound Machine, it's a CX50. The screen is cracked and the machine is not turning on."
MODEL = "gpt-4o"

# Steps:

init_diagnosis = initial_diagnosis(IMAGE_URL, USER_MESSAGE)

secondary_diagnosis = secondary_diagnosis(IMAGE_URL, init_diagnosis)

if json.loads(secondary_diagnosis)["remotely_solvable"] == True:
    print(create_diagnostic_report(secondary_diagnosis))

elif json.loads(secondary_diagnosis)["remotely_solvable"] == False:
    print(create_repair_plan(init_diagnosis, repair_plan.FLOWCHART_URL, repair_plan.QUESTION))  

else:
    print("No issues found.")


