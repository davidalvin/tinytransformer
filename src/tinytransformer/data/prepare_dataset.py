import os
import json
from datasets import load_dataset, load_from_disk
from tinytransformer.config.config import NUM_TRAIN_STORIES, TXT_PATH

RAW_DATASET_DIR = "data/tinystories_hf"  # disk cache path

SPLIT = "train"

def get_or_download_dataset(num_examples=None, split="train"):
    if os.path.exists(RAW_DATASET_DIR):
        print(f"📦 Loading cached dataset from {RAW_DATASET_DIR}")
        dataset = load_from_disk(RAW_DATASET_DIR)
    else:
        print("📥 Downloading TinyStories from Hugging Face...")
        dataset = load_dataset("roneneldan/TinyStories", split=split)
        if num_examples is not None:
            dataset = dataset.select(range(num_examples))
        dataset.save_to_disk(RAW_DATASET_DIR)
        print(f"✅ Saved subset to {RAW_DATASET_DIR}")
    return dataset

def convert_dataset_to_txt(dataset, txt_path):
    """Flatten to plain text with <|endoftext|> markers."""
    with open(txt_path, "w", encoding="utf-8") as f:
        for example in dataset:
            story = example["text"].strip()
            f.write(story + " <|endoftext|>\n")

def main():
    dataset = get_or_download_dataset(num_examples=NUM_TRAIN_STORIES, split=SPLIT)
    convert_dataset_to_txt(dataset, TXT_PATH)
    print(f"✅ Saved: {TXT_PATH}")

if __name__ == "__main__":
    main()
