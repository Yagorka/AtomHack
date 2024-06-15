from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import os

import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model_emb = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

def embeddings_text(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model_emb(**encoded_input)
    embeddings = model_output.pooler_output
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings


def reranker_part_chunk(sentences, question_embedding):
    context_embeddings = embeddings_text(sentences)
    res = torch.nn.functional.cosine_similarity(context_embeddings, question_embedding, dim=1)
    return res

def real_rerank(docs, query):
    question_embedding = embeddings_text([query])
    result_points = torch.zeros(len(docs))
    for i, doc in enumerate(docs):
        sentences = doc.page_content.split('. ')
        result_points[i] = reranker_part_chunk(sentences, question_embedding).max()
    return result_points   


embeddings = HuggingFaceEmbeddings(model_name="cointegrated/LaBSE-en-ru")



db = FAISS.load_local(
    os.path.join("..", "data", 'structured_vdb'),
    embeddings,
    allow_dangerous_deserialization = True
)

db2 = FAISS.load_local(
    os.path.join("..", "data", 'unstructured_vdb'),
    embeddings,
    allow_dangerous_deserialization = True
)

def search_best_from_structured(query):

    docs = db.similarity_search_with_score(query, k=5)
    norm_docs_from_excel = [i[0] for i in docs if i[1]<=0.658243]
    if len(norm_docs_from_excel)>1:
        print('ffghnfgnfg')
        result_points=real_rerank(norm_docs_from_excel, query)
        doc_real = norm_docs_from_excel[int(result_points.argmax())]
        return doc_real
    elif len(norm_docs_from_excel)==1:
        return docs[0]
    else:
        return None

def search_best_from_unstructured(query):

    docs = db2.similarity_search(query, k=5)
    if len(docs)>1:
        result_points=real_rerank(docs, query)
        doc_real = docs[int(result_points.argmax())]
        return doc_real