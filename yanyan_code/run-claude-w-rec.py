import pandas as pd
import json
from retry import retry

import anthropic


@retry(tries=5, delay=1, backoff=2)
def get_labels_and_reasons(model_input):
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.8,
        messages=[
            {"role": "user", "content": model_input}
        ]
    )
    return str(message.content)


def main():
    # example input
    with open("data_folder/prompts/prompt_w_recommendation.txt") as f:
        examples = f.read()

    # generate labels and reasons
    full_reasoning_labels = []
    for description in desc:
        few_shot_input = f"{examples} Description: {description}"
        print(few_shot_input)
        labels_and_reasons = get_labels_and_reasons(few_shot_input)
        print("Claude output:")
        print(labels_and_reasons)
        print()
        full_reasoning_labels.append(labels_and_reasons)
    with open("data_folder/output/claude_3_output_w_recommendation.json", 'w') as f:
        json.dump(full_reasoning_labels, f, indent=4)


if __name__ == "__main__":
    data_path = "data_folder/data/interaction.labeled.csv"

    # read the dataset
    sir_data = pd.read_csv(data_path)
    desc = sir_data['DESC'].tolist()

    # test first 5 rows
    # test_desc = desc[:5]

    # open the JSON file containing the API keys
    with open('data_folder//notes/claude_api.json') as config_file:
        config = json.load(config_file)

    client = anthropic.Anthropic(
        api_key=config['api_key']
    )

    main()
