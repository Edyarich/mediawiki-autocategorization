"""Inference ARTM model."""

import os
import shutil
import sys

sys.path.append("src")

import warnings

import pandas as pd

import artm
from artm_predict import predict
from collect_data import Dataset40x1tv
from preprocessing import filter_child_to_parent_categories, filtering_pipeline
from process_data import get_batch_vectorizer_and_dict

warnings.filterwarnings("ignore", category=DeprecationWarning)

WORKDIR = "temp_test"
MODEL_PATH = "artm/models/hier_modified"
FULL_PIPELINE_FLG = False

# Create working directories
os.makedirs(WORKDIR, exist_ok=True)
data_dir = os.path.join(WORKDIR, "data")
os.makedirs(data_dir, exist_ok=True)

# Copy vocabulary -- we won't create it
vocab_path = os.path.join(WORKDIR, "vocab_filtered.txt")
shutil.copyfile("./data/vocab_filtered.txt", vocab_path)

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

# Create batch vectorizer and vocabulary for ARTM inference
df = pd.read_parquet(df_path)
batch_vectorizer, dictionary = get_batch_vectorizer_and_dict(
    df,
    vocab_path=vocab_path,
    categories_hierarchy_path=hierarchy_path,
    target_folder=data_dir,
    test_flg=False,
)

# Load ARTM model
hier_model = artm.hARTM()
hier_model.load(MODEL_PATH)

# Predict categories and topics
prediction_path = os.path.join(WORKDIR, "prediction.csv")
df_pred = predict(hier_model, batch_vectorizer, MODEL_PATH, df, prediction_path)

for _, row in df_pred.sample(5).iterrows():
    print(row["doc_id"])
    print(row["categories"])
    print(row["generated_topics"])
    print()
