from openai import OpenAI
import os
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree
from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#from langsmith import traceable, get_current_run_tree

qdrant_client = QdrantClient(url='http://qdrant:6333')

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

@traceable(
    name='embed_query',
    run_type='embedding',
    metadata={
        'ls_provider': 'google',
        'ls_model_name': 'gemini-embedding-001',
    },
)
def get_embedding(text, model="gemini-embedding-001"):
    '''Gemini embedding has no usage tracking '''
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    # current_run = get_current_run_tree()
    # if current_run:
    #     current_run.metadata['usage_metadata'] = {
    #         'input_tokens': response.usage.prompt_tokens,
    #         'total_tokens': response.usage.total_tokens,
    #     }
    return response.data[0].embedding


@traceable(
    name='retrieve_data',
    run_type='retriever',
)
def retrieve_data(query, collection_name='amazon-items-collection-00', k=5):
    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context_scores = []
    retrieved_context_texts = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload['parent_asin'])
        retrieved_context_scores.append(result.score)
        retrieved_context_texts.append(result.payload['processed_description'])
        retrieved_context_ratings.append(result.payload['average_rating'])

    return {
        'retrieved_context_ids': retrieved_context_ids,
        'retrieved_context_scores': retrieved_context_scores,
        'retrieved_context_texts': retrieved_context_texts,
        'retrieved_context_ratings': retrieved_context_ratings
    }


@traceable(
    name='format_retrieved_context',
    run_type='prompt',
)
def process_context(retrieve_context):
    formatted_context = ''

    for id, chunk, rating in zip(retrieve_context['retrieved_context_ids'], retrieve_context['retrieved_context_texts'], retrieve_context['retrieved_context_ratings']):
        formatted_context += f"- Product ID: {id}, Product Rating: {rating}, Product Description: {chunk}\n"

    return formatted_context


@traceable(
    name='build_prompt',
    run_type='prompt',
)
def build_prompt(question, formatted_context):
    prompt = f"""
    You are a shopping assistant that can answer questions about the products in stock.

    You will be given a question and a list of context.

    Instructions:
    - Answer the question based on the context only.
    - Never use word context and refer to it as the available products.
    - Do not use markdown formatting

    Context:
    {formatted_context}

    Question:
    {question}
    """
    return prompt


@traceable(
    name='generate_answer',
    run_type='llm',
    metadata={
        'ls_provider': 'groq',
        'ls_model_name': 'openai/gpt-oss-120b',
    },
)
def generate_answer(prompt):
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        reasoning_effort="low",
        max_completion_tokens=1000,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'input_tokens': response.usage.prompt_tokens,
            'output_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
        }

    return response.choices[0].message.content

@traceable(name='rag_pipeline')
def rag_pipeline(question, topk=5):
    retrieve_context = retrieve_data(query=question, k=topk)
    formatted_context = process_context(retrieve_context)
    prompt = build_prompt(question=question, formatted_context=formatted_context)
    answer = generate_answer(prompt)

    return {
        'answer': answer,
        'question': question,
        'retrieved_context_ids': retrieve_context['retrieved_context_ids'],  
        'retrieved_context': retrieve_context['retrieved_context_texts'],    
    }