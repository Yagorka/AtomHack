import os
from transformers import GenerationConfig


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.vector_data import search_best_from_structured, search_best_from_unstructured

import time


model_id = 'IlyaGusev/saiga_llama3_8b'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # load_in_8bit=True,
    torch_dtype = torch.bfloat16,
    device_map = "auto"
)

model.eval()


generation_config = GenerationConfig.from_pretrained(model_id)
print(generation_config)

generation_config.max_new_tokens = 300
generation_config.temperature = 0.17
generation_config.top_k= 20
generation_config.top_p= 0.8

def get_llm_answer(query, chunks_join):
    user_prompt = '''Используй только следующий контекст, чтобы очень кратко ответить на вопрос в конце.
    Не пытайся выдумывать ответ.
    Контекст:
    ===========
    {chunks_join}
    ===========
    Вопрос:
    ===========
    {query}'''.format(chunks_join=chunks_join, query=query)
    
    SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
    RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
    
    prompt = f'''<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n{RESPONSE_TEMPLATE}'''
    
    def generate(model, tokenizer, prompt):
        data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(model.device) for k, v in data.items()}
        output_ids = model.generate(
            **data,
            generation_config=generation_config
        )[0]
        output_ids = output_ids[len(data["input_ids"][0]) :]
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        return output.strip()
    
    response = generate(model, tokenizer, prompt)
    
    return response

def QnA_with_LLM(query):
    res = ''
    excel_chunk = search_best_from_structured(input)
    if excel_chunk:
        print(excel_chunk)
        res += f'Решение аналогичного вопроса: {excel_chunk.metadata["Решение"]}'
        category = ' | '.join([f'{i}: {str(excel_chunk.metadata[i])}' for i in ["Аналитика 1", "Аналитика 2", "Аналитика 3"] ])
        res += f'\n {category}'
    
    pdf_chunk = search_best_from_unstructured(input)
    result = get_llm_answer(query, pdf_chunk.page_content).split('|im_end|>')[0]
    page = pdf_chunk.metadata["source"].split("/")[-1].split("_")[-1].split(".")[0]
    file_name = pdf_chunk.metadata['source'].split('/')[-2] + ".pdf"
    info = f'Страница: {page}, Исходный документ: {file_name}'
    res += '\n' + f'Ответ LLM: {result}' + '\n' + info
    return res

if __name__ == "__main__":
    import pandas as pd
    start_time = time.time()

    questions = pd.read_excel(os.path.join('..', 'data', 'q_by_pdf', 'вопросы.xlsx'), index_col=0)
    new_dict = {i: [] for i in questions.columns if i not in ['Категория']} | {'Мой полный ответ': [], 'Мой ответ': []}
    for i in range(len(questions)):
        temp_row = questions.iloc[i, :]
        new_dict['Вопрос'].append(temp_row['Вопрос'])
        res = ''
        pdf_chunk = search_best_from_unstructured(temp_row['Вопрос'])
        result = get_llm_answer(temp_row['Вопрос'], pdf_chunk.page_content).split('|im_end|>')[0]
        new_dict['Мой ответ'].append(result)
        new_dict['Правильный ответ'].append(temp_row['Правильный ответ'])
        page = pdf_chunk.metadata["source"].split("/")[-1].split("_")[-1].split(".")[0]
        new_dict['Страница'].append(page)
        file_name = pdf_chunk.metadata['source'].split('/')[-2] + ".pdf"
        new_dict['Файл'].append(file_name)
        info = f'Страница: {page}, Исходный документ: {file_name}'
        res += '\n ' + f'Ответ LLM: {result}' + '\n' + info
        new_dict['Мой полный ответ'].append(res)
    print("--- %s seconds ---" % (time.time() - start_time))
    df = pd.DataFrame(new_dict)
    df.to_csv(os.path.join('..', 'data', 'q_by_pdf', 'ответы на вопросы.csv'))


