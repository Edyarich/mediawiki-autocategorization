"""Functions for topic generation from its keywords."""

import re
from typing import List

import httpx
import tqdm

import artm

OLLAMA_CONFIG = {
    "keep_alive": "2m",
    "stream": False,
}

SYSTEM_PROMPT = (
    "You are an AI assistant tasked with generating precise and concise topic titles. "
    "Do not include explanations or reasoning—only provide the final title for each topic. "
    "Write answer in Russian language, but if there are foreign terms leave them as is."
)

SUMMARIZING_PROMPT = """
    You are an AI language model designed to analyze topics defined by keywords, bigrams, and categories,
    and then summarize each topic into a concise, descriptive name.
    Use the provided information (keywords, bigrams, and categories) to infer the central theme of each topic
    and generate a short, meaningful title. Follow these guidelines:

    1. Focus on the most relevant and representative keywords and bigrams to identify the primary subject of the topic.
    2. Use the categories to refine and contextualize the theme.
    3. Avoid overly generic titles; aim for specificity and clarity.
    4. Ensure the title reflects the core idea that unites the keywords, bigrams, and categories.

    Here is the input format:
    ```
    Topic: [topic_name]
    Words: [list of words]
    Bigrams: [list of bigrams]
    Categories: [list of categories]
    ```

    And here is an example of the desired output:
    ```
    Input:
    Topic: topic_15
    Words: ['архитектура', 'код', 'платформа', 'сервис', 'технология']
    Bigrams: ['хранилище_данные', 'облачный_сервис', 'распределённый_система', 'модуль_ядро', 'обработка_больший']
    Categories: ['программирование', 'программная_архитектура', 'распределенные_системы', 'виртуализация', 'paas']

    Output:
    "topic_15": "Архитектура программ и облачные сервисы"

    Input:
    Topic: topic_16
    Words: ['модель', 'текст', 'анализ', 'метод', 'алгоритм']
    Bigrams: ['ред_база', 'учебный_материал', 'субд_ред', 'вычислительный_комплекс', 'анализ_текст']
    Categories: ['алгоритмы', 'наука', 'machine_learning', 'natural_language_processing', 'data_analysis']

    Output:
    "topic_16": "Модели и алгоритмы анализа текста"

    Input:
    Topic: topic_17
    Words: ['проблема', 'требование', 'оценка', 'качество', 'заказчик']
    Bigrams: ['управление_требование', 'больший_количество', 'совместный_работа', 'оценка_качество', 'реализация_проект']
    Categories: ['бизнес_анализ', 'процесс_тестирования', 'управление_требованиями', 'дизайн', 'team_communication']

    Output:
    "topic_17": "Управление требованиями и дизайн"

    Input:
    Topic: topic_18
    Words: ['являться', 'любой', 'образ', 'позволять', 'часть']
    Bigrams: ['возможность_использование', 'высокий_уровень', 'данный_подход', 'код_проект', 'адресный_пространство']
    Categories: ['программирование', 'java', 'разработка_операционных_систем', 'моделирование_бизнес_процессов', 'доклад_со_стенограммой']

    Output:
    "topic_18": "Подходы в программировании и моделировании"

    Input:
    Topic: topic_19
    Words: ['компания', 'технический', 'российский', 'развитие', 'россия']
    Bigrams: ['команда_разработчик', 'ит_компания', 'технический_писатель', 'свой_проект', 'технический_долг']
    Categories: ['agile_process', 'государство_и_софт', 'документирование', 'спо_в_россии', 'планирование']

    Output:
    "topic_19": "Технические проекты в России"
    ```

    Now, process the following topics and generate titles for each:
"""


def run_prompt(
    endpoint: str,
    prompt: str,
    model: str = "gemma2",
    temperature: float = 0,
    seed: int = 123,
    **kwargs,
) -> httpx.Response:
    """_summary_

    Args:
        endpoint (str): точка подключения к API
        prompt (str): промпт для LLM
        model (str, optional): название модели. Defaults to "gemma2".
        temperature (float, optional): температура для генерации. Defaults to 0.
        seed (int, optional): сид для генерации. Defaults to 123.

    Returns:
        httpx.Response: ответ с endpoint
    """
    options_config = {"temperature": temperature, "seed": seed, **kwargs}

    combined_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    payload = {
        **OLLAMA_CONFIG,
        "model": model,
        "prompt": combined_prompt,
        "options": options_config,
    }

    response = httpx.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=300,
    )

    if response.status_code != 200:
        print("Error", response.status_code)
        assert False

    return response


def execute_prompt(
    prompt: str,
    endpoint: str = "http://127.0.0.1:11434/api/generate",
    model: str = "gemma2",
    **kwargs,
) -> str:
    """_summary_

    Args:
        prompt (str): промпт для LLM
        endpoint (str, optional): точка подключения к API.
            Defaults to 'http://127.0.0.1:11434/api/generate'.
        model (str, optional): название модели. Defaults to 'gemma2'.

    Returns:
        str: ответ от LLM
    """
    text = re.compile(r"#[^\n]*").sub("", prompt).strip()
    response = run_prompt(endpoint, text, model, **kwargs)
    response_content = response.json().get("response", "No response found.")
    return response_content


def postprocess_topic_name(topic_name: str) -> str:
    """
    Обрабатывает названия темы после генерации.

    Args:
        topic_name (str): сгенерированное название темы

    Returns:
        str: обработанное название темы
    """
    re_pattern = re.compile(r'"topic_\d+":\s*"([^"]+)"')
    return re.match(re_pattern, topic_name).groups()[0]


def get_top_tokens(model: artm.ARTM) -> List[str]:
    """
    Возвращает оформленный список топ-токенов для каждой темы.

    Args:
        model (artm.ARTM): объект ARTM

    Returns:
        List[str]: список топ-токенов для каждой темы
    """
    top_tokens1 = model.score_tracker["top-tokens-@word"].last_tokens
    top_tokens2 = model.score_tracker["top-tokens-@bigramm"].last_tokens
    top_tokens3 = model.score_tracker["top-tokens-@category"].last_tokens

    output = []

    for topic_name in model.topic_names:
        topic_descr = ""
        topic_descr += f"Topic: {topic_name}\n"
        topic_descr += f"Words: {None if topic_name not in top_tokens1 else top_tokens1[topic_name]}\n"
        topic_descr += f"Bigrams: {None if topic_name not in top_tokens2 else top_tokens2[topic_name]}\n"
        topic_descr += f"Categories: {None if topic_name not in top_tokens3 else top_tokens3[topic_name]}\n"
        output.append(topic_descr)

    return output


def generation_pipeline(model: artm.ARTM) -> List[str]:
    """
    Генерирует название темы по её ключевым словаи

    Args:
        model (artm.ARTM): объект ARTM

    Returns:
        List[str]: список названий тем
    """
    top_tokens_arr = get_top_tokens(model)
    titles_arr = []

    for top_tokens in tqdm.tqdm(top_tokens_arr):
        generated_topic = execute_prompt(
            SUMMARIZING_PROMPT + f"```\n{top_tokens}```", model="gemma2"
        )
        final_generated_topic = postprocess_topic_name(generated_topic)
        titles_arr.append(final_generated_topic)

    return titles_arr
