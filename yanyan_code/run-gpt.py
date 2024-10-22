import re
import json
from retry import retry

import openai
import pandas as pd


@retry(tries=5, delay=1, backoff=2)
def get_labels_and_reasons(model_input):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo-2024-04-09",
        messages=[
            {"role": "user", "content": model_input}
        ],
        temperature=0.8
    )
    return response.choices[0].message["content"]


def main():
    # example input
    with open("data_folder/prompts/prompt.txt") as f:
        examples = f.read()

    # generate labels and reasons
    full_reasoning_labels = []
    for description in test_desc:
        few_shot_input = f"{examples} Description: {description}"
        print(few_shot_input)
        labels_and_reasons = get_labels_and_reasons(few_shot_input)
        print("Chatgpt output:")
        print(labels_and_reasons)
        print()
        full_reasoning_labels.append(labels_and_reasons)
    with open("data_folder/output/gpt_4_output.json", 'w') as f:
        json.dump(full_reasoning_labels, f, indent=4)


if __name__ == "__main__":
    data_path = "data_folder/data/interaction.labeled.csv"

    # read the dataset
    sir_data = pd.read_csv(data_path)
    desc = sir_data['DESC'].tolist()

    # test first 5 rows
    test_desc = desc[:1]

    # get stored openai api key
    with open(
            'data_folder/notes/openai_api.json') as config_file:
        config = json.load(config_file)
    openai.api_key = config['api_key']
    openai.organization = config['organization']

    main()
