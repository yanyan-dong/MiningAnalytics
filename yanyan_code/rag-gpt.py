from sentence_transformers import SentenceTransformer, util
import torch
import glob
import os
import json
import pandas as pd
import openai
from retry import retry


@retry(tries=5, delay=1, backoff=2)
def get_generated_res(model_input):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo-2024-04-09",
        messages=[{"role": "user", "content": model_input}],
        temperature=0.8
    )
    return response.choices[0].message["content"]


# retrieve top k docuement
def retrieve_topk(query,
                  _corpus_embeddings,
                  _corpus,
                  top_k=10):
    # encode the user query to the query embedding
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # compute the cosine similarity between the query and the corpus
    cos_scores = util.cos_sim(query_embedding, _corpus_embeddings)[0]

    # get the top k most similar sentences
    top_results = torch.topk(cos_scores, k=top_k)

    _retrieved = []
    for score, idx in zip(top_results[0], top_results[1]):
        _retrieved.append(_corpus[idx])

    return _retrieved


def generate_response_with_few_shot():
    # load examples input for rag
    with open(
            "data_folder/prompts/prompt_w_recommendation_for_rag.txt") as f:
        examples = f.read()

    # generate response
    generated_text = []
    for description in desc:
        retrieved_documents = retrieve_topk(description, corpus_embeddings, corpus, top_k=5)

        # print out and check
        print(f"Job Description: {description}")
        print(f"Retrieved documents:")
        for doc in retrieved_documents:
            print(doc)

        # set up few shot input
        few_shot_input = (
            f"Here are some knowledge related to mining safety management: \n{' '.join(retrieved_documents)}\n"
            f"{examples}\n"
            f"Description: {description}\n"
        )

        # check input
        print(few_shot_input)

        # call get_generated_res() function
        generated_res = get_generated_res(few_shot_input)

        print("Chatgpt output:")
        print(generated_res)
        generated_text.append(generated_res)
        print("-" * 60)

    with open("data_folder/output/rag_output_gpt_4.json",
              'w') as f:
        json.dump(generated_text, f, indent=4)


if __name__ == "__main__":
    # glob all the text files in the directory
    files = glob.glob("data_folder/rag_data_txt/*.txt")

    # get the text from the files
    corpus = []
    for file in files:
        # get the file name
        file_name = os.path.basename(file)

        # remove the file extension
        file_name = os.path.splitext(file_name)[0]
        with open(file, "r") as f:
            document_text = f.read()

            # split the documents in to 200 word chunks
            document_text = document_text.split()
            for i in range(0, len(document_text), 50):
                corpus.append(f"file name: {file_name} chunk {i}\n{' '.join(document_text[i:i + 50])}")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # load SIR dataset
    SIR_data_path = "data_folder/data/interaction.labeled.csv"

    # read the dataset
    SIR_data = pd.read_csv(SIR_data_path)
    desc = SIR_data['DESC'].tolist()

    # testing
    # test_desc = desc[:10]

    # get stored openai api key
    with open(
            'data_folder/notes/openai_api.json') as config_file:
        config = json.load(config_file)
    openai.api_key = config['api_key']
    openai.organization = config['organization']

    # call function
    generate_response_with_few_shot()
