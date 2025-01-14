"""Functions for training and inference with ARTM."""

import itertools
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import artm
from constants import ARTM_MODALITIES
from process_data import transform_category
from utils import iou_metric


def calc_words_co_occurrence(
    bigramm_table: np.ndarray,
    bigramm_vocabulary: List[str],
    vocabulary: List[str],
    output_file: str,
):
    """
    Вычисляет со-встречаемость слов в корпусе и сохраняет её в VowpalWabbit-формате.

    Args:
        bigramm_table (np.ndarray): матрица частотности биграмм
        bigramm_vocabulary (List[str]): словарь биграмм
        vocabulary (List[str]): словарь термов
        output_file (str): _description_
    """
    bigramm_vocab_to_id = {v: i for i, v in enumerate(bigramm_vocabulary)}
    bigramm_vocabulary = set(bigramm_vocabulary)
    bigramm_stats = bigramm_table.sum(axis=0)

    fd = open(output_file, "w", encoding="utf-8")

    for word1 in vocabulary:
        output = []
        for word2 in vocabulary:
            if word1 == word2:
                continue

            for bigramm in [f"{word1}_{word2}", f"{word2}_{word1}"]:
                if bigramm in bigramm_vocabulary:
                    output.append(
                        f"|@word {word1} |@word {word2} {bigramm_stats[bigramm_vocab_to_id[bigramm]]}"
                    )

        if len(output) > 0:
            for line in output:
                fd.write(line + "\n")

    fd.close()


def extract_categories_pairs(categories: List[str]) -> List[str]:
    """
    Составляет все возможные пары из списка категорий.
    Предварительно, все категории преобразуются в более читаемый вид.

    Args:
        categories (List[str]): список категорий

    Returns:
        List[str]: список всех пар категорий
    """
    all_pairs = list(itertools.combinations(categories, 2))
    all_pairs = list(
        map(
            lambda x: f"{transform_category(x[0])}_{transform_category(x[1])}",
            all_pairs,
        )
    )
    return all_pairs


def calc_categories_co_occurrence(
    counter: Dict[Any, int],
    pairs_vocabulary: List[str],
    vocabulary: List[str],
    output_file: str,
):
    """
    Вычисляет со-встречаемость категорий в корпусе и сохраняет её в VowpalWabbit-формате.

    Args:
        counter (Dict[Any, int]): список частот встречаемости пар категорий
        pairs_vocabulary (List[str]): словарь пар категорий
        vocabulary (List[str]): словарь категорий
        output_file (str): выходной файл
    """
    pairs_vocabulary = set(pairs_vocabulary)
    bigramm_stats = counter

    fd = open(output_file, "w", encoding="utf-8")

    for word1 in vocabulary:
        output = []
        for word2 in vocabulary:
            if word1 == word2:
                continue

            for bigramm in [f"{word1}_{word2}", f"{word2}_{word1}"]:
                if bigramm in pairs_vocabulary:
                    output.append(f"|@category {word1} |@category {word2} {bigramm_stats[bigramm]}")

        if len(output) > 0:
            for line in output:
                fd.write(line + "\n")

    fd.close()


def overlap_between_levels(
    level1: artm.ARTM,
    level2: artm.ARTM,
    score_names: List[str] = ("top-tokens-@word", "top-tokens-@bigramm", "top-tokens-@category"),
) -> float:
    """
    Вычисляет долю пересечения ключевых слов между различными моделями.
    Для различных уровней одной иерархической модели: меньше -- лучше.

    Args:
        level1 (artm.ARTM): _description_
        level2 (artm.ARTM): _description_
        score_names (List[str], optional): _description_.
            Defaults to ("top-tokens-@word", "top-tokens-@bigramm", "top-tokens-@category").

    Returns:
        float: средний IoU между ключевыми словами различных моделей
    """
    top_tokens1 = [level1.score_tracker[score_nm].last_tokens for score_nm in score_names]
    top_tokens2 = [level2.score_tracker[score_nm].last_tokens for score_nm in score_names]

    top_tokens1_all_topics = [
        list(map(lambda x: top_tokens1i[x], level1.topic_names)) for top_tokens1i in top_tokens1
    ]
    top_tokens2_all_topics = [
        list(map(lambda x: top_tokens2i[x], level2.topic_names)) for top_tokens2i in top_tokens2
    ]

    overlap_arr = [
        iou_metric(list(itertools.chain(*x)), list(itertools.chain(*y)))
        for x, y in zip(top_tokens1_all_topics, top_tokens2_all_topics)
    ]
    return np.mean(overlap_arr)


def topics_cosine_distance(model: artm.ARTM) -> float:
    """
    Вычисляет косинусовую расстояние между темами.

    Args:
        model (artm.ARTM): объект ARTM

    Returns:
        float: косинусовая расстояние между темами
    """
    phi = model.get_phi().values
    cosine_mat = np.dot(phi.T, phi)
    mask = np.triu(np.ones_like(cosine_mat, dtype=bool), k=1)
    return 1 - np.mean(cosine_mat[mask])


def mean_parents_count(model: artm.hARTM) -> float:
    """
    Вычисляет среднее число родителей для каждой темы.

    Args:
        model (artm.hARTM): объект ARTM

    Returns:
        float: среднее число родителей для каждой темы
    """
    psi = model.get_psi()
    psi_threshold = psi.values.max(axis=1).min()
    parent_counts = (psi > psi_threshold).sum(axis=1).mean()
    return parent_counts


def classification_iou(model: artm.ARTM, target: List[List[str]], cutoff: float = 0.1) -> float:
    """
    Вычисляет качество классификации иерархической модели по IoU

    Args:
        model (artm.ARTM): объект ARTM
        target (List[List[str]]): список целевых категорий
        cutoff (float, optional): вероятность включения категории в предсказание модели.
            Defaults to 0.1.

    Returns:
        float: IoU по обучающей выборке
    """
    # model.transform doesn't work properly, so I'm going to implement it implicitly
    phi_mat = model.get_phi()
    theta_mat = model.get_theta()
    noncategory_indices = list(filter(lambda x: x[0] != "@category", phi_mat.index.values))
    phi_mat.drop(index=noncategory_indices, inplace=True)

    categories_prob = phi_mat @ theta_mat
    categories_pred = categories_prob > cutoff
    categories_list = np.array(list(map(lambda x: x[1], categories_prob.index.values)))

    prediction = []
    for doc_id in range(categories_pred.shape[1]):
        prediction.append(categories_list[categories_pred.iloc[:, doc_id]].tolist())

    return np.mean([iou_metric(trg, pred) for trg, pred in zip(target, prediction)])


def degenerated_topics_ratio(level: artm.ARTM, cutoff: float = 0.95) -> float:
    """
    Вычисляет соотношение вырожденных тем к общему количеству тем.

    Args:
        level (artm.ARTM): объект ARTM
        cutoff (float, optional): порог для определения вырожденности темы.
            Defaults to 0.95.

    Returns:
        float: соотношение вырожденных тем к общему количеству тем
    """
    phi_mat_sorted = np.sort(level.get_phi().values, axis=0)
    return np.mean(phi_mat_sorted[-1:].sum(axis=0) >= cutoff)


def small_topics_ratio(level: artm.ARTM, min_topic_size: int = 5) -> float:
    """
    Вычисляет соотношение маленьких тем к общему количеству тем.

    Args:
        level (artm.ARTM): объект ARTM
        min_topic_size (int, optional): минимальный размер темы. Defaults to 5.

    Returns:
        float: соотношение маленьких тем к общему количеству тем
    """
    theta_mat = level.get_theta().values
    doc_to_topic = theta_mat.argmax(axis=0)
    _, topic_size_distr = np.unique(doc_to_topic, return_counts=True)
    return np.mean(topic_size_distr < min_topic_size)


def add_scores_and_regularizers(model: artm.ARTM, level_id: Optional[int] = None, **kwargs):
    """
    Добавляет счётчики и регуляризаторы в ARTM.

    Args:
        model (artm.ARTM): объект ARTM
        level_id (Optional[int], optional): id модели в иерархии. Defaults to None.
        **kwargs: параметры регуляризации
    """
    class_ids = list(model.class_ids.keys())
    thematic_topic_names = model.topic_names
    decorrelator_class_ids = class_ids[:2]  # drop @category from decorrelator class_ids

    coherence_dicts = kwargs.get("coherence_dicts", dict(zip(class_ids, [None] * len(class_ids))))
    decorrelator_weights = kwargs.get("decorrelator_weights", [0] * int(level_id + 1))
    sparse_phi_weights = kwargs.get("sparse_phi_weights", [0] * int(level_id + 1))
    sparse_theta_weights = kwargs.get("sparse_theta_weights", [0] * int(level_id + 1))
    label_phi_weights = kwargs.get("label_phi_weights", [0] * int(level_id + 1))
    hier_psi_weights = kwargs.get("hier_psi_weights", [0] * int(level_id + 1))
    coherence_phi_weights = kwargs.get("coherence_phi_weights", [0] * int(level_id + 1))
    topic_select_weights = kwargs.get("topic_select_weights", [0] * int(level_id + 1))

    model.scores.add(artm.PerplexityScore(name="perplexity"))
    model.scores.add(artm.SparsityThetaScore(name="sparsity_theta_score"))

    for class_id in class_ids:
        model.scores.add(
            artm.SparsityPhiScore(name=f"sparsity_phi_score_{class_id}", class_id=class_id)
        )
        model.scores.add(
            artm.TopTokensScore(
                name=f"top-tokens-{class_id}",
                num_tokens=5,
                class_id=class_id,
                dictionary=coherence_dicts[class_id],
            )
        )

        if level_id > 0:
            model.scores.add(
                artm.TopicKernelScore(
                    name=f"contrast_score_{class_id}",
                    probability_mass_threshold=0.25,
                    class_id=class_id,
                )
            )

    model.regularizers.add(
        artm.SmoothSparsePhiRegularizer(
            name="sparse_phi_regularizer", class_ids=class_ids, topic_names=thematic_topic_names
        )
    )
    model.regularizers.add(
        artm.SmoothSparseThetaRegularizer(
            name="sparse_theta_regularizer", topic_names=thematic_topic_names
        )
    )
    model.regularizers.add(
        artm.DecorrelatorPhiRegularizer(
            name="decorrelator_phi_regularizer",
            class_ids=decorrelator_class_ids,
            topic_names=thematic_topic_names,
        )
    )
    model.regularizers.add(
        artm.LabelRegularizationPhiRegularizer(
            name="label_phi_regularizer", class_ids="@category", topic_names=thematic_topic_names
        )
    )
    model.regularizers.add(
        artm.ImproveCoherencePhiRegularizer(
            name="coherence_phi_regularizer",
            class_ids="@word",
            dictionary=coherence_dicts.get("@word"),
            topic_names=thematic_topic_names,
        )
    )

    model.regularizers["decorrelator_phi_regularizer"].tau = decorrelator_weights[level_id]
    model.regularizers["sparse_phi_regularizer"].tau = sparse_phi_weights[level_id]
    model.regularizers["sparse_theta_regularizer"].tau = sparse_theta_weights[level_id]
    model.regularizers["label_phi_regularizer"].tau = label_phi_weights[level_id]
    model.regularizers["coherence_phi_regularizer"].tau = coherence_phi_weights[level_id]

    if level_id > 0:
        model.regularizers.add(
            artm.HierarchySparsingThetaRegularizer(
                name="hier_psi_regularizer",
            )
        )
        model.regularizers["hier_psi_regularizer"].tau = hier_psi_weights[level_id]

        model.regularizers.add(
            artm.TopicSelectionThetaRegularizer(
                name="topic_select_regularizer", topic_names=thematic_topic_names
            )
        )
        model.regularizers["topic_select_regularizer"].tau = topic_select_weights[level_id]


def train_model(
    model: artm.ARTM,
    bv: artm.BatchVectorizer,
    num_epochs: int,
    print_step: int,
    logger_func: Optional[Callable] = None,
    tau_schedulers: Optional[Dict[str, List[float]]] = None,
):
    """
    Обучает ARTM в оффлайн режиме.

    Args:
        model (artm.ARTM): объект ARTM
        bv (artm.BatchVectorizer): объект BatchVectorizer
        num_epochs (int): количество эпох
        print_step (int): шаг вывода сообщений
        logger_func (Callable, optional): функция для вывода сообщений. Defaults to None.
        tau_schedulers (Dict[str, List[float]], optional): список значений tau для регуляризации.
            Defaults to None.
    """

    def update_tau(model, tau_schedulers, epoch):
        if tau_schedulers is None:
            return

        for regularizer_name, tau_arr in tau_schedulers.items():
            model.regularizers[regularizer_name].tau = tau_arr[epoch]

    for i in range(num_epochs):
        update_tau(model, tau_schedulers, i)
        model.fit_offline(bv, num_collection_passes=1)

        if i % print_step == 0 or i + 1 == num_epochs and logger_func is not None:
            logger_func(model, i)


def log_training_step(model: artm.ARTM, it: int, train_target: Optional[List[List[str]]] = None):
    """
    Выводит сообщения о прогрессе обучения.

    Args:
        model (artm.ARTM): объект ARTM
        it (int): текущая эпоха
        train_target (List[List[str]], optional): список целевых тем.
            Defaults to None.
    """
    nan_value = "Not defined"
    pplx = round(model.score_tracker["perplexity"].last_value, 2)

    sparse_phi_1 = model.score_tracker["sparsity_phi_score_@word"].last_value
    sparse_phi_2 = model.score_tracker["sparsity_phi_score_@bigramm"].last_value
    sparse_phi_3 = model.score_tracker["sparsity_phi_score_@category"].last_value
    sparse_phi = round(np.mean([sparse_phi_1, sparse_phi_2, sparse_phi_3]), 2)
    sparse_theta = round(model.score_tracker["sparsity_theta_score"].last_value, 2)

    coherence_1 = model.score_tracker["top-tokens-@word"].last_average_coherence
    coherence_3 = model.score_tracker["top-tokens-@category"].last_average_coherence
    coherence = round(coherence_1 * 0.75 + coherence_3 * 0.25, 2)

    if train_target:
        classification_score = round(classification_iou(model, train_target), 3)
    else:
        classification_score = nan_value

    cosine_dissimilarity = round(topics_cosine_distance(model), 3)
    avg_parents = (
        nan_value if not hasattr(model, "get_psi") else round(mean_parents_count(model), 2)
    )

    contrast_score_1 = (
        None
        if "contrast_score_@word" not in model.score_tracker
        else model.score_tracker["contrast_score_@word"].last_average_contrast
    )
    contrast_score_2 = (
        None
        if "contrast_score_@bigramm" not in model.score_tracker
        else model.score_tracker["contrast_score_@bigramm"].last_average_contrast
    )
    contrast_score_3 = (
        None
        if "contrast_score_@category" not in model.score_tracker
        else model.score_tracker["contrast_score_@category"].last_average_contrast
    )
    contrast_score = (
        nan_value
        if contrast_score_1 is None
        else round(np.mean([contrast_score_1, contrast_score_2, contrast_score_3]), 2)
    )

    avg_degenerated_topics = round(degenerated_topics_ratio(model), 2)
    avg_small_topics = round(small_topics_ratio(model), 2)

    print(
        f"Iter #{it}, perplexity: {pplx}, sparse_phi: {sparse_phi}, sparse_theta: {sparse_theta}, "
        f"coherence: {coherence}, classification: {classification_score}, parents: {avg_parents}, "
        f"contrast: {contrast_score}, distance between topics: {cosine_dissimilarity}, "
        f"degenerated topics: {avg_degenerated_topics}, small topics: {avg_small_topics}"
    )


def instantiate_model(
    dictionary: artm.Dictionary,
    batch_vectorizer: artm.BatchVectorizer,
    class_weights: List[float],
    topic_sizes: List[int],
    parent_weights: List[float],
    train_level_func: Callable = train_model,
    train_log_func: Callable = log_training_step,
    tmp_files_path: str = "./",
    num_document_passes: int = 3,
    class_ids: List[str] = ARTM_MODALITIES,
    k_levels: int = 3,
    num_epochs: int = 50,
    print_step: int = 5,
    **kwargs,
) -> artm.hARTM:
    """
    Создаёт и обучает иерархическую модель.

    Args:
        dictionary (artm.Dictionary): объект Dictionary
        batch_vectorizer (artm.BatchVectorizer): объект BatchVectorizer
        class_weights (List[float]): веса модальностей
        topic_sizes (List[int]): размеры тем
        parent_weights (List[float]): веса родительских моделей
        train_level_func (Callable, optional): функция для обучения модели.
            Defaults to train_model.
        train_log_func (Callable, optional): функция для логирования обучения.
            Defaults to log_training_step.
        tmp_files_path (str, optional): путь для сохранения моделью ARTM различных файлов.
            Defaults to "./".
        num_document_passes (int, optional): количество проходов по корпусу документов.
            Defaults to 3.
        class_ids (List[str], optional): список модальностей. Defaults to ARTM_MODALITIES.
        k_levels (int, optional): количество уровней. Defaults to 3.
        num_epochs (int, optional): число эпох обучения каждого уровня. Defaults to 50.
        print_step (int, optional): шаг вывода логов. Defaults to 5.

    Returns:
        artm.hARTM: объект hARTM
    """
    assert len(class_ids) == len(class_weights)
    assert len(topic_sizes) == len(parent_weights) == k_levels

    hier = artm.hARTM(
        dictionary=dictionary,
        num_document_passes=num_document_passes,
        cache_theta=True,
        tmp_files_path=tmp_files_path,
    )
    hier.class_ids = dict(zip(class_ids, class_weights))

    log_func = partial(train_log_func, train_target=kwargs.get("target_categories"))

    for level_id in range(k_levels):
        level = hier.add_level(
            num_topics=topic_sizes[level_id],
            parent_level_weight=parent_weights[level_id],
        )
        level.initialize(dictionary)
        add_scores_and_regularizers(level, level_id, **kwargs)

        print(f"Training level {level_id}")
        tau_schedulers = None

        if level_id > 0:
            session_1_length, session_2_length = num_epochs // 2, (num_epochs + 1) // 2
            topic_select_tau = level.regularizers["decorrelator_phi_regularizer"].tau
            decorrelator_tau = level.regularizers["topic_select_regularizer"].tau

            tau_schedulers = {
                "topic_select_regularizer": [0] * session_1_length
                + [topic_select_tau] * session_2_length,
                "decorrelator_phi_regularizer": [decorrelator_tau] * session_1_length
                + [0] * session_2_length,
            }

        train_level_func(
            level,
            batch_vectorizer,
            num_epochs,
            print_step,
            log_func,
            tau_schedulers,
        )

    return hier
