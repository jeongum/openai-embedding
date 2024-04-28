from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import os
from openai import OpenAI

app = Flask(__name__)

load_dotenv()
es = Elasticsearch("http://localhost:9200")  # Alternatively use `api_key` instead of `basic_auth`

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def search_similarity(user_embedding):
    similar_docs = es.search(
        index='faq-index',
        body={
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                        "params": {"query_vector": user_embedding}
                    }
                }
            },
            "_source": ["question", "answer"],
            "size": 1
        }
    )

    # 가장 유사한 document
    hit_document = similar_docs['hits']['hits'][0]

    # document의 유사도
    score = hit_document['_score']
    print(score)

    # 가장 유사한 document의 답변 출력
    return hit_document['_source']['answer']


def index_document(question, answer, embedding):
    es.index(
        index='faq-index',
        body={
            'question': question,
            'answer': answer,
            'content_vector': embedding,
        }
    )


def chat_with_bot(user_message):
    # 사용자 메시지 임베딩 생성
    user_embedding = get_embedding(user_message)

    # Elasticsearch에서 유사한 문서 검색
    similarity = search_similarity(user_embedding)

    return similarity


def embed_and_store_cases(question, answer):
    # 임베딩 생성
    embedding = get_embedding(question)
    # 임베딩 저장
    index_document(question, answer, embedding)


def create_es_index():
    index_mapping = {
        "properties": {
            "question": {
                "type": "text",
            },
            "answer": {
                "type": "text",
            },
            "content_vector": {
                "type": "dense_vector",
                "dims": 1536,
                "index": "true",
                "similarity": "cosine"
            }
        }
    }
    es.indices.create(index="faq-index", mappings=index_mapping)


@app.route('/answer', methods=['POST'])
def get_answer():
    user_message = request.json['user_message']

    # 챗봇 로직 호출
    response_messages = chat_with_bot(user_message)

    return jsonify({"answer": response_messages})


@app.route('/store', methods=['POST'])
def store_knowledge():
    question = request.json['question']
    answer = request.json['answer']

    embed_and_store_cases(question, answer)

    return jsonify({"result": "OK"})


@app.route('/create', methods=['POST'])
def create_index():
    create_es_index()

    return jsonify({"result": "OK"})


if __name__ == '__main__':
    app.run()

