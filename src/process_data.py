"""Functions for training and inference with ARTM."""

import os
import pickle
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import artm
from constants import ARTM_MODALITIES, SEP_TOKEN


def extract_bigrams(tokens: List[str], stopwords: List[str]) -> List[str]:
    """
    Составляет корректные биграммы из списка токенов.
    Корректная биграмма -- биграмма, в которой нет стоп-слов.

    Args:
        tokens (List[str]): список токенов
        stopwords (List[str]): стоп-слова в виде списка токенов

    Returns:
        List[str]: список биграмм
    """
    bigrams = []
    for i in range(len(tokens) - 1):
        if tokens[i] not in stopwords and tokens[i + 1] not in stopwords:
            bigrams.append(f"{tokens[i]}_{tokens[i + 1]}")
    return bigrams


def get_word_table(df: pd.DataFrame, vocabulary_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Возвращает таблицу частотности слов и словарь в виде списка.

    Args:
        df (pd.DataFrame): датафрейм со столбцом 'text', содержащий список слов
        vocabulary_path (str): путь к файлу с словарём

    Returns:
        Tuple[np.ndarray, List[str]]: таблица частотности и словарь
    """
    with open(vocabulary_path, "r", encoding="utf-8") as fd:
        vocabulary = fd.read().splitlines()

    vocabulary_to_id = {token: i for i, token in enumerate(vocabulary)}
    word_doc_matrix = np.zeros((len(vocabulary), len(df)), dtype=np.uint32)

    for i, row in df.iterrows():
        for word in row["text"]:
            if word in vocabulary_to_id:
                word_doc_matrix[vocabulary_to_id[word], i] += 1

    return word_doc_matrix.T, vocabulary


def build_count_vectorizer_on_bigrams(
    documents: List[List[str]], vocab: List[str], stopwords: List[str]
) -> np.ndarray:
    """
    Вычисляет таблицу частотности биграмм.

    Args:
        documents (List[List[str]]): список списков с токенами
        vocab (List[str]): словарь с словами
        stopwords (List[str]): список стоп-слов

    Returns:
        np.ndarray: таблица частотности биграмм
    """
    vocab_to_id = {v: i for i, v in enumerate(vocab)}
    cv_matrix = np.zeros((len(documents), len(vocab)), dtype=np.uint32)
    for i, doc in enumerate(documents):
        for j, _ in enumerate(doc):
            if j + 1 == len(doc):
                break

            if doc[j] not in stopwords and doc[j + 1] not in stopwords:
                bigramm = f"{doc[j]}_{doc[j + 1]}"
                if bigramm in vocab_to_id:
                    cv_matrix[i, vocab_to_id[bigramm]] += 1

    return cv_matrix


def get_bigramm_table(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Возвращает таблицу частотности биграмм и словарь биграмм в виде списка.

    Args:
        df (pd.DataFrame): датафрейм со столбцом 'text', содержащий список слов

    Returns:
        Tuple[np.ndarray, List[str]]: таблица частотности и словарь
    """
    bigram_counter = Counter()
    stopwords = [SEP_TOKEN]

    for tokens in df["text"]:
        bigram_counter.update(extract_bigrams(tokens, stopwords))

    bigramm_vocabulary = list(
        map(lambda x: x[0], filter(lambda x: x[1] >= 5, bigram_counter.items()))
    )
    bigramm_table = build_count_vectorizer_on_bigrams(df["text"], bigramm_vocabulary, stopwords)
    return bigramm_table, bigramm_vocabulary


def transform_category(category: str) -> str:
    """
    Преобразование категории: приведение к нижнему регистру, удаление спецсимволов.

    Args:
        category (str): название категории

    Returns:
        str: преобразованное название категории
    """
    category = category.lower()
    if category == "c++":
        return category

    category = re.sub(r"[ /-]", "_", category)
    category = re.sub(r"[^а-яa-z0-9_]", "", category)
    category = category.strip("_")
    return category


def build_count_vectorizer_on_categories(
    documents: List[List[str]],
    categories_vocab: List[str],
    child_to_parent_map: Dict[str, List[str]],
) -> np.ndarray:
    """
    Вычисляет таблицу частотности категорий.
    Учитываются как исходные категории, так и все их родители.

    Args:
        documents (List[List[str]]): список списков с категориями
        categories_vocab (List[str]): словарь с категориями
        child_to_parent_map (Dict[str, List[str]]): словарь с дочерними и родительскими категориями

    Returns:
        np.ndarray: таблица частотности категорий
    """
    ctgry_to_id = {v: i for i, v in enumerate(categories_vocab)}
    cv_matrix = np.zeros((len(documents), len(categories_vocab)), dtype=np.uint32)

    for i, doc in enumerate(documents):
        for _, ctgry in enumerate(doc):
            if ctgry in categories_vocab:
                cv_matrix[i, ctgry_to_id[ctgry]] += 1

            for parent_ctgry in child_to_parent_map[ctgry]:
                if parent_ctgry in categories_vocab:
                    cv_matrix[i, ctgry_to_id[parent_ctgry]] += 1

    return cv_matrix


def tf_matrices_to_vw(
    tf_matrices: List[np.ndarray],
    vocabularies: List[List[str]],
    namespaces: List[str],
    output_file: str,
):
    """
    Конвертируем матрицы частотности в формат VowpalWabbit.

    Args:
        tf_matrices (List[np.ndarray]): список матриц частотности
        vocabularies (List[List[str]]): список словарей
        namespaces (List[str]): список модальностей
        output_file (str): выходной файл
    """
    num_documents = tf_matrices[0].shape[0]

    with open(output_file, "w", encoding="utf-8") as f:
        for doc_idx in range(num_documents):
            vw_line = []
            for i, (ns, matrix, vocab) in enumerate(zip(namespaces, tf_matrices, vocabularies)):
                features = matrix[doc_idx]
                tokens = [f"{vocab[idx]}:{val}" for idx, val in enumerate(features) if val > 0]

                if tokens:
                    output_str = f"|{ns} " + " ".join(tokens)
                    if i == 0:
                        output_str = f"doc{doc_idx} " + output_str

                    vw_line.append(output_str)

            f.write(" ".join(vw_line) + "\n")


def vocab_to_vw(vocabularies: List[List[str]], namespaces: List[str], output_file: str):
    """
    Записывает словарь в формате VowpalWabbit.

    Args:
        vocabularies (List[List[str]]): список словарей
        namespaces (List[str]): список пространств имён
        output_file (str): путь к файлу для записи
    """
    with open(output_file, "w", encoding="utf-8") as fd:
        for vocab, namespace in zip(vocabularies, namespaces):
            for word in vocab:
                fd.write(f"{word} {namespace}\n")


def get_category_table(
    df: pd.DataFrame, categories_hierarchy_path: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Возвращает таблицу частотности категорий и словарь в виде списка.

    Args:
        df (pd.DataFrame): датафрейм со столбцом 'categories', содержащий список категорий
        categories_hierarchy_path (str): путь к файлу с иерархией категорий

    Returns:
        Tuple[np.ndarray, List[str]]: таблица частотности и словарь
    """
    with open(categories_hierarchy_path, "rb") as fd:
        child_to_parent_categories = pickle.load(fd)

    categories_vocab = list(child_to_parent_categories.keys())
    categories_table = build_count_vectorizer_on_categories(
        df["categories"], categories_vocab, child_to_parent_categories
    )
    categories_vocab = list(map(transform_category, categories_vocab))

    return categories_table, categories_vocab


def get_batch_vectorizer_and_dict(
    df: pd.DataFrame,
    vocab_path: str,
    categories_hierarchy_path: str,
    target_folder: str,
    test_flg: bool = False,
) -> Tuple[artm.BatchVectorizer, artm.Dictionary]:
    """
    Возвращает BatchVectorizer и Dictionary для ARTM.

    Args:
        df (pd.DataFrame): датафрейм со столбцами 'text' и 'categories',
            содержащий список слов и список категорий соответственно
        vocab_path (str): путь к файлу с словарём
        categories_hierarchy_path (str): путь к файлу с иерархией категорий
        target_folder (str): путь к папке для сохранения файлов
        test_flg (bool): флаг для тестирования; если истиннен, то убирается модальность категорий.
            Defaults to False.

    Returns:
        Tuple[artm.BatchVectorizer, artm.Dictionary]: BatchVectorizer и Dictionary
    """
    artm_collection_path = os.path.join(target_folder, "multimodal_collection.txt")
    artm_vocab_path = os.path.join(target_folder, "vocabulary.txt")
    artm_dictionary_path = os.path.join(target_folder, "dictionary.txt")

    word_doc_matrix, vocabulary = get_word_table(df, vocab_path)
    bigramm_table, bigramm_vocabulary = get_bigramm_table(df)

    if test_flg:
        tf_matrices = [word_doc_matrix, bigramm_table]
        vocabularies = [vocabulary, bigramm_vocabulary]
    else:
        categories_table, categories_vocab = get_category_table(df, categories_hierarchy_path)
        tf_matrices = [word_doc_matrix, bigramm_table, categories_table]
        vocabularies = [vocabulary, bigramm_vocabulary, categories_vocab]\
    
    tf_matrices_to_vw(tf_matrices, vocabularies, ARTM_MODALITIES, artm_collection_path)
    vocab_to_vw(vocabularies, ARTM_MODALITIES, artm_vocab_path)

    batch_vectorizer = artm.BatchVectorizer(
        data_path=artm_collection_path,
        data_format="vowpal_wabbit",
        target_folder=target_folder,
        batch_size=1500,
    )
    dictionary = batch_vectorizer.dictionary
    dictionary.filter(min_df=3)
    dictionary.save_text(artm_dictionary_path)

    return batch_vectorizer, dictionary
