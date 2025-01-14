""" Function to assign categories to a document based on the ARTM model."""

import os
import pickle

import numpy as np
import pandas as pd

import artm


def predict_topics(
    hier_model: artm.hARTM,
    bv_test: artm.BatchVectorizer,
    titles_path: str,
    df: pd.DataFrame,
    k_topics: int = 3,
    **kwargs
) -> pd.DataFrame:
    """
    Предсказыавет сгенерированные названия тем

    Args:
        hier_model (artm.hARTM): обученная модель hARTM
        bv_test (artm.BatchVectorizer): тестовый объект BatchVectorizer
        titles_path (str): путь к сгенерированным названиям тем
        df (pd.DataFrame): датафрейм с id документов, содержит столбец 'doc_id'
        k_topics (int, optional): кол-во сгенерированных тем. Defaults to 3.

    Returns:
        pd.DataFrame: датасет с предсказанными темами
    """
    k_levels = hier_model.num_levels
    theta_arr = []

    for lvl in range(k_levels):
        theta_lvl = hier_model.get_level(lvl).transform(bv_test)
        theta_lvl.set_index(
            [[lvl + 1] * len(theta_lvl), np.arange(len(theta_lvl)), theta_lvl.index], inplace=True
        )
        theta_arr.append(theta_lvl)

    df_merged = pd.concat(theta_arr, axis=0)
    relevant_topic_ids = df_merged.values.argsort(axis=0)[-k_topics:].T

    with open(os.path.join(titles_path, "titles_lvl1.pkl"), "rb") as fd:
        generated_titles_lvl1 = pickle.load(fd)

    with open(os.path.join(titles_path, "titles_lvl2.pkl"), "rb") as fd:
        generated_titles_lvl2 = pickle.load(fd)

    with open(os.path.join(titles_path, "titles_lvl3.pkl"), "rb") as fd:
        generated_titles_lvl3 = pickle.load(fd)

    generated_titles = np.array(
        generated_titles_lvl1 + generated_titles_lvl2 + generated_titles_lvl3
    )
    relevant_topics = [
        [generated_titles[idx] for idx in topic_ids] for topic_ids in relevant_topic_ids
    ]

    output_df = df[["doc_id"]].copy()
    output_df["generated_topics"] = relevant_topics
    return output_df


def predict_categories(
    hier_model: artm.hARTM,
    bv_test: artm.BatchVectorizer,
    df: pd.DataFrame,
    pred_cutoff: float = 0.1,
    **kwargs
) -> pd.DataFrame:
    """
    Предсказыавет названия категорий

    Args:
        hier_model (artm.hARTM): обученная модель hARTM
        bv_test (artm.BatchVectorizer): тестовый объект BatchVectorizer
        df (pd.DataFrame): датафрейм с id документов, содержит столбец 'doc_id'
        pred_cutoff (float, optional): порог на p(ctgry | doc). Defaults to 0.1.

    Returns:
        pd.DataFrame: датасет с предсказанными категориями
    """
    last_artm_model = hier_model.get_level(hier_model.num_levels - 1)

    phi_mat = last_artm_model.get_phi()
    theta_mat = last_artm_model.transform(bv_test)
    noncategory_indices = list(filter(lambda x: x[0] != "@category", phi_mat.index.values))
    phi_mat.drop(index=noncategory_indices, inplace=True)

    categories_pred = (phi_mat @ theta_mat) > pred_cutoff
    categories_list = np.array(list(map(lambda x: x[1], categories_pred.index.values)))

    prediction = []
    for doc_id in range(categories_pred.shape[1]):
        prediction.append(categories_list[categories_pred.iloc[:, doc_id]].tolist())

    output_df = df[["doc_id"]].copy()
    output_df["categories"] = prediction
    return output_df


def predict(
    hier_model: artm.hARTM,
    bv_test: artm.BatchVectorizer,
    titles_path: str,
    df: pd.DataFrame,
    save_path: str = None,
    **pred_kwargs
) -> pd.DataFrame:
    """
    Предсказыавет названия категорий и сгенерированных ранее тем

    Args:
        hier_model (artm.hARTM): обученная модель hARTM
        bv_test (artm.BatchVectorizer): тестовый объект BatchVectorizer
        titles_path (str): путь к сгенерированным названиям тем
        df (pd.DataFrame): датафрейм с id документов, содержит столбец 'doc_id'
        save_path (str, optional): путь для сохранения предсказаний. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    topics_df = predict_topics(hier_model, bv_test, titles_path, df, **pred_kwargs)
    categories_df = predict_categories(hier_model, bv_test, df, **pred_kwargs)

    output_df = pd.merge(topics_df, categories_df, on="doc_id")
    if save_path:
        output_df.to_csv(save_path, index=False)

    return output_df
