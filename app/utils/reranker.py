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

