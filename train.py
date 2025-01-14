"""Train ARTM model."""

import os
import pickle
import shutil
import sys

sys.path.append("src")

import warnings

import pandas as pd

from artm_models import instantiate_model
from artm_predict import predict
from collect_data import Dataset40x1tv
from generate_categories import generation_pipeline
from preprocessing import filter_child_to_parent_categories, filtering_pipeline
from process_data import get_batch_vectorizer_and_dict, transform_category

warnings.filterwarnings("ignore", category=DeprecationWarning)

WORKDIR = "temp_train"
FULL_PIPELINE_FLG = False

# Create working directories
data_dir = os.path.join(WORKDIR, "data")
model_dir = os.path.join(WORKDIR, "artm_model")

os.makedirs(WORKDIR, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Copy vocabulary -- we won't create it
vocab_path = os.path.join(WORKDIR, "vocab_filtered.txt")
shutil.copyfile("data/vocab_filtered.txt", vocab_path)

# Parse data from mediawiki
data_path = os.path.join(WORKDIR, "0x1tv-dataset.pickle")
if FULL_PIPELINE_FLG:
    dataset_obj = Dataset40x1tv()
    dataset_obj.process(data_path)

# Filter data, save fataframe to 'df_path'
df_path = os.path.join(WORKDIR, "data.parquet")
if FULL_PIPELINE_FLG:
    filtering_pipeline(data_path, df_path)

# Filter child_to_parent categories
hierarchy_path = os.path.join(WORKDIR, "child_to_parent_categories.pkl")
if FULL_PIPELINE_FLG:
    filter_child_to_parent_categories(data_path, hierarchy_path)

# Create batch vectorizer and vocabulary for ARTM training
df = pd.read_parquet(df_path)

target_categories = list(
    map(lambda x: [transform_category(el) for el in x], df["categories"].values)
)

batch_vectorizer, dictionary = get_batch_vectorizer_and_dict(
    df,
    vocab_path=vocab_path,
    categories_hierarchy_path=hierarchy_path,
    target_folder=data_dir,
    test_flg=False,
)

# Train ARTM model
hier_new = instantiate_model(
    dictionary,
    batch_vectorizer,
    class_weights=[1.0, 3.0, 10.0],
    topic_sizes=[15, 50, 150],
    parent_weights=[1, 1, 10],
    tmp_files_path=WORKDIR,
    decorrelator_weights=[1e5, 0, 0],
    sparse_phi_weights=[0.5, -0.5, -0.3],
    sparse_theta_weights=[-0.75, -0.75, -0.5],
    label_phi_weights=[1, 10, 100],
    hier_psi_weights=[0, 10, 100],
    topic_select_weights=[0, 10, 100],
    k_levels=3,
    num_epochs=40,
    print_step=5,
    target_categories=target_categories,
)

# Drop all files if the directory is not empty
if os.listdir(model_dir):
    shutil.rmtree(model_dir)
    os.makedirs(model_dir)

hier_new.save(model_dir)

# Generate topic names
if False:
    generated_titles_lvl1 = generation_pipeline(hier_new.get_level(0))
    generated_titles_lvl2 = generation_pipeline(hier_new.get_level(1))
    generated_titles_lvl3 = generation_pipeline(hier_new.get_level(2))

    with open(os.path.join(model_dir, "titles_lvl1.pkl"), "wb") as fd:
        pickle.dump(generated_titles_lvl1, fd)

    with open(os.path.join(model_dir, "titles_lvl2.pkl"), "wb") as fd:
        pickle.dump(generated_titles_lvl2, fd)

    with open(os.path.join(model_dir, "titles_lvl3.pkl"), "wb") as fd:
        pickle.dump(generated_titles_lvl3, fd)

    # Predict categories and topics
    prediction_path = os.path.join(WORKDIR, "prediction.csv")
    df_pred = predict(hier_new, batch_vectorizer, model_dir, df, prediction_path)

    for _, row in df_pred.sample(5).iterrows():
        print(row["doc_id"])
        print(row["categories"])
        print(row["generated_topics"])
        print()
