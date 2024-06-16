
import datetime

import os
from transformers import GenerationConfig


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.vector_data import search_best_from_structured, search_best_from_unstructured

import time

from config import token


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
    
    SYSTEM_PROMPT = "Ты система поддержки пользователей компании Росатом. Ты разговариваешь с людьми и помогаешь им."
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
    excel_chunk = search_best_from_structured(query)
    if excel_chunk:
        print(excel_chunk)
        res += f'Решение аналогичного вопроса: {excel_chunk[0].metadata["Решение"]}'
        category = ' | '.join([f'{i}: {str(excel_chunk[0].metadata[i])}' for i in ["Аналитика 1", "Аналитика 2", "Аналитика 3"] ])
        res += f'\n {category}'
    
    pdf_chunk = search_best_from_unstructured(query)
    result = get_llm_answer(query, pdf_chunk.page_content).split('|im_end|>')[0]
    page = pdf_chunk.metadata["source"].split("/")[-1].split("_")[-1].split(".")[0]
    file_name = pdf_chunk.metadata['source'].split('/')[-2] + ".pdf"
    info = f'Страница: {page}, Исходный документ: {file_name}'
    res += '\n' + f'Ответ LLM: {result}' + '\n' + info
    return res


# Вместо BOT TOKEN HERE нужно вставить токен вашего бота, полученный у @BotFather
BOT_TOKEN = token






from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message


# Создаем объекты бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


# Этот хэндлер будет срабатывать на команду "/start"
@dp.message(Command(commands=["start"]))
async def process_start_command(message: Message):
    await message.answer('Привет!\nЯ помощник службы поддержки! Задай мне вопрос!\n')


# Этот хэндлер будет срабатывать на команду "/help"
@dp.message(Command(commands=['help']))
async def process_help_command(message: Message):
    await message.answer(
        'Напиши мне что-нибудь и в ответ '
        'и я пришлю тебе ответ'
    )


# Этот хэндлер будет срабатывать на любые ваши текстовые сообщения,
# кроме команд "/start" и "/help"
@dp.message()
async def send_echo(message: Message):
    inputs = str(message.text)
    res = QnA_with_LLM(inputs)
    await message.reply(text=res)


if __name__ == '__main__':
    dp.run_polling(bot)
