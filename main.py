import time
# CLI
import argparse
# EMBEDDING VECTOR DATABASE
from langchain_chroma import Chroma
# LLM 
from langchain_core.prompts import ChatPromptTemplate
# Embeddings
from create_db import EMBEDDINGS, CHUNK_SIZES, BASE_CHROMA_PATH


SCORE_THRESHOLD = 0.6
NUMBER_OF_RESULTS = 4

PROMPT_TEMPLATE = """
Answer the question based only on the context:
context: {context}
question: {query}
"""

PROMPT_TEMPLATE_FR = """
Répondez soigneusement et en détail à la question basée uniquement sur le contexte:
contexte: {context}
question: {query}
"""

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    return args.query_text

def get_results(query):
    start = time.time()

    results = []
    for CHUNK_SIZE in CHUNK_SIZES:
        CHROMA_PATH = f"{BASE_CHROMA_PATH}_{CHUNK_SIZE}"
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=EMBEDDINGS)
        res = db.similarity_search_with_relevance_scores(query, k=NUMBER_OF_RESULTS)

        if not len(res) or res[0][1] < SCORE_THRESHOLD:
            print(f"No results found above threshold {SCORE_THRESHOLD} in {CHROMA_PATH}")
            context_text = "\n\n\n" + "\n\n---\n\n".join(["\n---\n".join([doc.page_content, str(score)]) for doc, score in res])
            print(f"Context:\n{context_text}")
            pass 

        results.extend(res)
    
    print(f"querying in {time.time() - start:.2f} seconds") 
    
    return results


def create_prompt(results, query):
    # Create prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query)

    # LLM MistralAI: limited usage 
    # llm = ChatMistralAI(model="mistral-large-latest")
    # response = llm.invoke(prompt)

    messages = [
        {"role": "user", "content": prompt},
    ]

    return messages


def response_llm(messages, results):
    
    ## USING HUGGINGFACE PIPELINE
    # os.environ["HF_TOKEN"] = ""
    # pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct")
    # response = pipe(messages)
    # print("pipeline")
    # print(response)

    # USING OLLAMA LIBRARY
    # response = ollama.chat(model="llama3.1", messages=messages)['message']['content']
    # print(res)

    # USING LANGHCAIN INTEGRATION WITH OLLAMA
    from langchain_community.llms import Ollama
    llm = Ollama(model="llama3.1")
    response = llm.invoke(messages)

    sources = [(doc.metadata.get("id", None), _score) for doc, _score in results]
    formatted_response_sources = f"Response: {response} \n\n #Sources: {sources}"
    formatted_response = f"Response: {response}"
    return formatted_response


def main():
    query = cli()
    results = get_results(query)
    if not len(results):
        print("No results found above threshold")
        return
    prompt = create_prompt(results, query)
    formatted_response = response_llm(prompt, results)
    print("formatted")
    print(formatted_response) 

if __name__ == "__main__":
    main()