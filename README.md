# AtomicHack
# ИИ Ассистент "Капитал"

### Разработан в рамках [AtomicHack](https://atomichack.ru/) командой "hacken4221".

### Live версия доступна [тут](https://t.me/rosatom_support_bot)

## Use Cases

### Поиск по неструктурированным данным (вопросы из документации в виде pdf)


Поиск информации по заданной тематике на основе неструктурированных данных (текста).

----

### Поиск по структурированным данным (схожие вопросы на основе истории обращений)


Поиск информации по заданной тематике на основе структурированных данных (таблицы).

----

## Архитектура решения

<!-- <img src="./resources/img/architecture.jpg" alt="Архитектура решения" width="700"/> -->

* Движок ассистента разработан на основе LangChain (две векторные бд FAISS для pdf и excel соответсвтенно). 
* В качестве LLM используется opensource saiga_llama3_8b, это модель семейства llama3 обученная на русском языке и весящая +- 17 гб GPU 
* cointegrated/LaBSE-en-ru используется для получения 'embedding в RAG подходе и в качестве reranker (уточнение контекстных чанков).
* Для чтения pdf используется pytesseract (для изображений) + PyPDF2 (для обычных pdf страниц).
 
### Структура проекта

```
├── app # основная директория проекта
│   ├── utils # содержит утилиты для работы проекта
│   ├── main_app.py # тг бот
│   ├── chat_bot.py # файл ответов на вопросы в формате excel
│   ├── app.ipynb # demo бота с gradio
├── data # содержит данные и БД для проекта, а также тестовые вопросы
├── README.md
├── requirements.txt
└── resources # ресурсы проекта
```

# How to run? Запуск решения

## Development

0. Install requirements

```
pip install -r requirements.txt
```
1. Добавьте config.py -> app
```
token = 'ваш тг токен'
```
2. Запустите вашего тг бота перейдя в директорию app
```
python main_app.py
```