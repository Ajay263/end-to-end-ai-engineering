import os
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from langsmith import Client
from ragas.llms import LangchainLLMWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from api.agents.retrieval_generation import rag_pipeline, get_embedding

ls_client = Client()
qdrant_client = QdrantClient(url='http://qdrant:6333')

# --- Ragas evaluator LLM (Groq) ---
ragas_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )
)

# --- Ragas evaluator embeddings (Gemini) ---
class GeminiEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return get_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [get_embedding(t) for t in texts]

ragas_embeddings = GeminiEmbeddings()


# --- Evaluators ---

def ragas_context_precision_id_based(run, example):
    retrieved_ids = [str(i) for i in run.outputs['retrieved_context_ids']]
    reference_ids = {str(i) for i in example.outputs['reference_context_ids']}

    if not retrieved_ids:
        return {"results": [{"key": "context_precision", "score": 0.0}]}

    hits = sum(1 for rid in retrieved_ids if rid in reference_ids)
    score = hits / len(retrieved_ids)
    return {"results": [{"key": "context_precision", "score": score}]}


def ragas_context_recall_id_based(run, example):
    retrieved_ids = {str(i) for i in run.outputs['retrieved_context_ids']}
    reference_ids = {str(i) for i in example.outputs['reference_context_ids']}

    if not reference_ids:
        return {"results": [{"key": "context_recall", "score": 0.0}]}

    hits = sum(1 for rid in reference_ids if rid in retrieved_ids)
    score = hits / len(reference_ids)
    return {"results": [{"key": "context_recall", "score": score}]}


def ragas_faithfulness(run, example):
    sample = SingleTurnSample(
        user_input=run.outputs['question'],
        response=run.outputs['answer'],
        retrieved_contexts=run.outputs['retrieved_context'],
    )

    scorer = Faithfulness(llm=ragas_llm)
    score = scorer.single_turn_score(sample)
    return {"results": [{"key": "faithfulness", "score": float(score)}]}


def ragas_answer_relevancy(run, example):
    sample = SingleTurnSample(
        user_input=run.outputs['question'],
        response=run.outputs['answer'],
        retrieved_contexts=run.outputs['retrieved_context'],
    )

    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    score = scorer.single_turn_score(sample)
    return {"results": [{"key": "answer_relevancy", "score": float(score)}]}


# --- Run evaluation ---

results = ls_client.evaluate(
    lambda x: rag_pipeline(x['question'], qdrant_client),
    data='rag-eval-dataset',
    evaluators=[
        ragas_context_precision_id_based,
        ragas_context_recall_id_based,
        ragas_faithfulness,
        ragas_answer_relevancy,
    ],
    experiment_prefix='retriever',
    max_concurrency=5
)