"""Functions to view and vizualize ARTM predictions."""

from collections import Counter

import numpy as np
import plotly.graph_objects as go

import artm


def child_to_parent(
    parent_level: artm.ARTM, child_level: artm.ARTM, phi_path: str = "artm/models/phi1.batch"
) -> np.ndarray:
    """
    Соотносит детскую тему с родительской темой.

    Args:
        parent_level (artm.ARTM): родительский объект ARTM
        child_level (artm.ARTM): дочерний объект ARTM
        phi_path (str, optional): путь к матрицам Phi (слова х темы).
            Defaults to "artm/models/phi1.batch".

    Returns:
        np.ndarray: массив размера (k_child_topics,) со значениями {0,1,...,k_parent_topics-1}
            в каждом ячейке
    """
    batch = artm.messages.Batch()

    with open(phi_path, "rb") as f:
        batch.ParseFromString(f.read())

        n_tw = np.zeros(len(parent_level.topic_names))

        for i, item in enumerate(batch.item):  # pylint: disable=E1101:no-member
            for _, token_weight in zip(item.field[0].token_id, item.field[0].token_weight):
                n_tw[i] += token_weight

        psi = child_level.get_psi()
        n_t1_t0 = np.array(psi) * n_tw
        psi_bayes = (n_t1_t0 / n_t1_t0.sum(axis=1)[:, np.newaxis]).T

        return np.argmax(psi_bayes, axis=0)


def print_top_tokens(model: artm.ARTM):
    """
    Выводит список наиболее релевантных токенов для каждой темы.

    Args:
        model (artm.ARTM): объект ARTM
    """
    top_tokens1 = model.score_tracker["top-tokens-@word"].last_tokens
    top_tokens2 = model.score_tracker["top-tokens-@bigramm"].last_tokens
    top_tokens3 = model.score_tracker["top-tokens-@category"].last_tokens

    for topic_name in model.topic_names:
        print("Topic:", topic_name)
        print("Words:", None if topic_name not in top_tokens1 else top_tokens1[topic_name])
        print(
            "Bigrams:",
            None if topic_name not in top_tokens2 else top_tokens2[topic_name],
        )
        print(
            "Categories:",
            None if topic_name not in top_tokens3 else top_tokens3[topic_name],
        )
        print("=" * 80)


def vizualize_topics(model: artm.hARTM, phi_path: str):
    """
    Визуализирует размеры тем и соотношения между ними.
    Пока реализовано для уровней 0 и 1 иерархической модели.

    Args:
        model (artm.hARTM): объект hARTM
        phi_path (str): путь к матрицам Phi (слова х темы)
    """
    theta_0 = model.get_level(0).get_theta()
    theta_1 = model.get_level(1).get_theta()

    doc_to_topic_0 = theta_0.values.argmax(axis=0)
    doc_to_topic_1 = theta_1.values.argmax(axis=0)

    parent_child_weights = Counter(zip(doc_to_topic_0, doc_to_topic_1))
    indexes_child = child_to_parent(model.get_level(0), model.get_level(1), phi_path)

    level0_names = model.get_level(0).topic_names
    level1_names = model.get_level(1).topic_names
    node_labels = level0_names + level1_names

    sources, targets, values = [], [], []
    for child_idx, parent_idx in enumerate(indexes_child):
        sources.append(parent_idx)
        targets.append(len(level0_names) + child_idx)
        values.append(parent_child_weights[(parent_idx, child_idx)])

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=10,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
            ),
        )
    )

    fig.update_layout(
        title_text="Multilevel Topic Graph Visualization",
        font_size=12,
        width=900,
        height=1600,
    )

    fig.show()
