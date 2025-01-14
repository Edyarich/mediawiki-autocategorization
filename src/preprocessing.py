"""Functions for preprocessing the text."""

import os
import pickle
import re
from collections import Counter
from typing import Dict, List, Optional, T

import nltk
import pandas as pd
import pymorphy2
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from constants import (
    ALL_CATEGORIES_WITHOUT_NAMES,
    FILE_EXTENSIONS,
    LATEX_MARKDOWN_COMMANDS,
    SEP_TOKEN,
)
from stop_words import (
    ENG_STOP_WORDS,
    RUSSIAN_NAMES,
    RUSSIAN_STOP_WORDS,
    RUSSIAN_SURNAMES,
)

NLTK_QUITE_FLG = False

nltk.download("punkt_tab", download_dir="nltk_data/punkt_tab", quiet=NLTK_QUITE_FLG)
nltk.download("punkt", download_dir="nltk_data/punkt", quiet=NLTK_QUITE_FLG)
nltk.download("stopwords", download_dir="nltk_data/stopwords", quiet=NLTK_QUITE_FLG)
nltk.download("wordnet", download_dir="nltk_data/wordnet", quiet=NLTK_QUITE_FLG)
nltk.download("omw-1.4", download_dir="nltk_data/omw-1.4", quiet=NLTK_QUITE_FLG)

NLTK_PATH = "../nltk_data/"
for subdir in os.scandir(NLTK_PATH):
    if subdir.is_dir():
        nltk.data.path.append(subdir.path)

DOC_FIELDS = ["title", "annotation", "thesis"]
THESIS_SECTIONS = ["Thesis", "Тезисы", "Расширенные тезисы"]
ALL_SECTIONS = ["Аннотация"] + THESIS_SECTIONS


def get_min_position(*positions, default=0) -> float:
    """
    Вспомогательная функция, которая вычисляет минимальное неотрицательное число из входного списка

    Args:
        default (int, optional): Значение по умолчанию,
            будет возвращено в случае, если все числа отрицательны. Defaults to 0.

    Returns:
        float: минимальное неотрицательное число из входного списка
    """
    return min([pos for pos in positions if pos >= 0], default=default)


def extract_section_text(data: str, section_name: str) -> str:
    """
    Извлекает подстроку с соответствующим разделом

    Args:
        data (str): входной текст
        section_name (str): название секции

    Returns:
        str: содержимое секции
    """
    if section_name == "Аннотация":
        blockquote_pattern = r"<blockquote>(.*?)</blockquote>"
        match = re.search(blockquote_pattern, data, re.DOTALL)
        return match.group(1).strip() if match else None

    section_pattern = rf"==\s*{re.escape(section_name)}\s*=="
    section_start = re.search(section_pattern, data)

    if not section_start:
        return None

    position = section_start.end()

    # Look for stop tokens and the start of next sections
    stop_tokens = ["{{----}}", "{{LinksSection}}", "== Примечания и отзывы =="]
    ends = [data.find(token, position) for token in stop_tokens]
    end_position = get_min_position(*ends, default=len(data))

    if end_position != len(data):
        return data[position:end_position].strip()

    return data[position:].strip()


def extract_sections(data: str) -> Dict[str, str]:
    """
    Извлекает содержимое всех секций из текста

    Args:
        data (str): текст

    Returns:
        Dict[str, str]: содержимое всех секций в виде словаря
    """
    return {section: extract_section_text(data, section) for section in ALL_SECTIONS}


def coalesce(dct: dict, keys: List[T]) -> Optional[T]:
    """
    Выбирает первое значение, которое не None из списка ключей

    Args:
        dct (dict): словарь
        keys (List[T]): список ключей

    Returns:
        Optional[T]: первое значение, которое не None из списка ключей
    """
    for key in keys:
        if dct[key] is not None:
            return dct[key]
    return None


def speaker_detect(text: str) -> List[str]:
    """
    Определяет распознаваемых спикеров из текста

    Args:
        text (str): текст

    Returns:
        List[str]: список распознаваемых спикеров
    """
    speakers = re.findall("(?<={{Speaker\|).*(?=}})", text)
    preproc_speakers = []
    for speaker in speakers:
        preproc_speakers.extend(speaker.split("|"))
    return preproc_speakers


def clear_categories(sample: List[str]) -> List[str]:
    """
    Очищает список категорий, удаляя подстроку `Категория:

    Args:
        sample (List[str]): список категорий

    Returns:
        List[str]: список очищенных категорий
    """
    return [re.sub("Категория:", "", el) for el in sample]


def parse_document_with_categories(doc: dict) -> dict:
    """
    Парсит документ и возвращает словарь с разобранными данными

    Args:
        doc (dict): документ c ключами 'title', 'text' и 'categories'

    Returns:
        dict: словарь с разобранными данными, содержит ключи DOC_FIELDS и 'categories'
    """
    parsed_doc = dict()
    parsed_doc["title"] = doc["title"]
    parsed_doc["speakers"] = speaker_detect(doc["text"])

    sections_data = extract_sections(doc["text"])
    parsed_doc["annotation"] = sections_data["Аннотация"]
    parsed_doc["thesis"] = coalesce(
        sections_data, ["Thesis", "Тезисы", "Расширенные тезисы"]
    )

    raw_categories = clear_categories(doc["categories"])
    parsed_doc["categories"] = list(
        filter(
            lambda x: x in ALL_CATEGORIES_WITHOUT_NAMES,
            raw_categories,
        )
    )
    return parsed_doc


def is_russian(word: str) -> bool:
    """
    Проверяет, является ли слово русским

    Args:
        word (str): слово

    Returns:
        bool: булев флаг того, русское ли слово
    """
    return bool(re.search("[а-яА-Я]", word))


def clean_text(text: str, use_lemmatization: bool = True) -> List[str]:
    """
    Очистка текста от лишних символов и слов:
        - удаление пунктуации
        - удаление ссылок
        - удаление всех HTML тегов
        - удаление файловых расширений и всех слов, их содержащих
        - удаление команд из $\LaTeX$ и Markdown
        - удаление чисел как отдельных токенов (если в смычке с тексом -- то ОК)
        - удаление стоп-слов
        - приведение к нижнему регистру
        - лемматизация

    Args:
        text (str): текст
        use_lemmatization (bool, optional): флаг включения лемматизации. Defaults to True.

    Returns:
        List[str]: отфильтрованный текст в виде списка слов.
    """
    if text is None or len(text) == 0:
        return ""

    latex_markdown_pattern = re.compile(
        "|".join([re.escape(cmd) for cmd in LATEX_MARKDOWN_COMMANDS]), re.IGNORECASE
    )
    file_extensions_pattern = re.compile(
        r"\b\w+\.(?:" + "|".join(FILE_EXTENSIONS) + r")\b", re.IGNORECASE
    )
    english_lemmatizer = WordNetLemmatizer()
    morph = pymorphy2.MorphAnalyzer()

    # Remove all HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove all URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove LaTeX commands (e.g., \begin, \end)
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    # Remove punctuation using regex to cover more symbols
    text = re.sub(r"[^\w\s]", "", text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Convert to lower case
    tokens = [token.lower() for token in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words("english") + stopwords.words("russian"))
    stop_words = stop_words.union(
        ENG_STOP_WORDS | RUSSIAN_STOP_WORDS | RUSSIAN_NAMES | RUSSIAN_SURNAMES
    )
    tokens = [word for word in tokens if word not in stop_words]

    # Drop standalone digits (keep tokens with at least one letter)
    tokens = [word for word in tokens if re.search("[A-Za-zа-яА-Я]", word)]

    cleaned_tokens = []
    for word in tokens:
        if latex_markdown_pattern.search(word):
            continue

        if file_extensions_pattern.search(word):
            continue

        if any(ext in word for ext in FILE_EXTENSIONS):
            continue

        if use_lemmatization:
            lemmatized = (
                morph.parse(word)[0].normal_form
                if is_russian(word)
                else english_lemmatizer.lemmatize(word)
            )
            cleaned_tokens.append(lemmatized)
        else:
            cleaned_tokens.append(word)

    return " ".join(cleaned_tokens)


def filter_child_to_parent_categories(
    input_path: str = "../data/0x1tv-dataset.pickle",
    output_path: str = "../data/child_to_parent_categories.pkl",
) -> None:
    """
    Фильтрует иерархию "дочерняя категория: список родительских категорий"

    Args:
        input_path (str, optional): путь к файлу с иерархией.
            Defaults to "../data/0x1tv-dataset.pickle".
        output_path (str, optional): путь для сохранения фильтрованной иерархии.
            Defaults to "../data/child_to_parent_categories.pkl".
    """
    with open(input_path, "rb") as fd:
        data = pickle.load(fd)

    assert isinstance(data, dict), "Input data must be a dictionary"
    assert "categories" in data, "Input data must contain a key 'categories'"

    child_to_parent_categories = {
        item["title"]: clear_categories(item["categories"])
        for item in data["categories"]
    }
    child_to_parent_categories_filt = {
        k: v
        for k, v in child_to_parent_categories.items()
        if k in ALL_CATEGORIES_WITHOUT_NAMES
    }
    not_presented_categories = list(
        set(ALL_CATEGORIES_WITHOUT_NAMES) - set(child_to_parent_categories.keys())
    )

    for ctgry in not_presented_categories:
        child_to_parent_categories_filt[ctgry] = []

    with open(output_path, "wb") as fd:
        pickle.dump(child_to_parent_categories_filt, fd)


def clean_and_merge_document(doc: dict) -> dict:
    """
    Очищает и объединяет данные документа

    Args:
        doc (dict): словарь с данными документа;
            содержит ключи DOC_FIELDS и 'categories'

    Returns:
        dict: словарь с ключами 'text' и 'categories'
    """
    cleaned_texts = []

    # Clean each field and collect non-empty results
    for field in DOC_FIELDS:
        field_text = doc.get(field, "")
        cleaned_tokens = clean_text(field_text)

        if cleaned_tokens:
            cleaned_texts.append(cleaned_tokens)

    # Join with <sep> token
    result_text = f" {SEP_TOKEN} ".join(cleaned_texts).strip()

    result = {
        "text": result_text.split(),
        "categories": doc.get("categories", []),
    }

    return result


def filter_dataframe(df: pd.DataFrame):
    """
    Фильтрует датафрейм по количеству слов в тексте

    Args:
        df (pd.DataFrame): датафрейм со столбцом 'text', содержащий список слов
    """
    df.drop(index=df.index[df.text.apply(lambda x: len(x) <= 3)], inplace=True)
    df.reset_index(inplace=True, drop=False)


def filtering_pipeline(
    input_path: str = "../data/0x1tv-dataset.pickle",
    output_path: str = "../data/processed_data_v2.parquet",
    vocab_path: Optional[str] = None,
):
    """
    Процесс извлечения и фильтрации текста из датасета

    Args:
        input_path (str, optional): путь к файлу с датасетом.
            Defaults to "../data/0x1tv-dataset.pickle".
        output_path (str, optional): файл для сохранения обработанного датасета в формате parquet.
            Defaults to "../data/processed_data_v2.parquet".
        vocab_path (Optional[str], optional): файл для сохранения словаря.
            Defaults to None.
    """
    with open(input_path, "rb") as fd:
        data = pickle.load(fd)

    data_raw = [parse_document_with_categories(doc) for doc in data["articles"]]
    processed_data = [clean_and_merge_document(doc) for doc in data_raw]

    words_counter = Counter()
    for clear_doc in processed_data:
        words_counter.update(clear_doc["text"])

    all_words_sorted = words_counter.most_common()

    # Выкинем все слова с частотой встречи < 3 или длинной 1
    words_to_drop = list(filter(lambda x: x[1] < 3 or len(x[0]) <= 1, all_words_sorted))
    words_to_drop = list(map(lambda x: x[0], words_to_drop))
    words_to_drop = set(words_to_drop)

    if vocab_path:
        filtered_vocab = list(
            set(list(map(lambda x: x[0], all_words_sorted))) - set(words_to_drop)
        )

        with open(vocab_path, "w", encoding="utf-8") as fd:
            fd.write("\n".join(filtered_vocab))

    processed_data_v2 = [
        {
            "text": list(filter(lambda x: x not in words_to_drop, doc["text"])),
            "categories": doc["categories"],
        }
        for doc in processed_data
    ]
    df_2 = pd.DataFrame(processed_data_v2)
    df_2["doc_id"] = [doc["title"] for doc in data["articles"]]
    filter_dataframe(df_2)
    df_2.to_parquet(output_path, index=False)
