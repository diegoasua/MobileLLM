# Import required libraries
import os
import json
from typing import Dict, Iterable
import time

import datasets as hf_datasets
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from tqdm.notebook import tqdm

OUTPUT_DIR = "basepath/1"  # Directory to save output files
TOKENIZER_DIR = "configs/125M"  # Directory containing tokenizer files
# set higher than total samples. Training code only supports 1 file/node
SAMPLES_PER_FILE = 100_000_000  # Number of samples per output file
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-100BT"
TOTAL_TOKENS = 100_000_000_000  # 100B tokens
TOKENS_PER_SAMPLE = 2048
BATCH_SIZE = 512
NUM_WORKERS = 40

class ConcatTokensDataset(IterableDataset):
    def __init__(
        self,
        hf_dataset: hf_datasets.IterableDataset,
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        buffer = []
        for sample in self.hf_dataset:
            encoded = self.tokenizer(sample['text'], truncation=False, padding=False, add_special_tokens=False)
            iids = encoded['input_ids']
            buffer.extend(iids)
            while len(buffer) >= self.max_length:
                concat_sample = buffer[:self.max_length]
                buffer = buffer[self.max_length:]
                yield {
                    'tokens': np.array(concat_sample, dtype=np.int64).tobytes()
                }

def main():
    # Calculate expected totals
    total_samples = TOTAL_TOKENS // TOKENS_PER_SAMPLE
    expected_files = (total_samples + SAMPLES_PER_FILE - 1) // SAMPLES_PER_FILE
    total_iterations = total_samples // BATCH_SIZE

    print(f"Expected total samples: {total_samples}")
    print(f"Expected number of files: {expected_files}")
    print(f"Expected number of iterations: {total_iterations}")
    print(f"Estimated time: {total_iterations * 3.5 / 3600:.2f} to {total_iterations * 4.5 / 3600:.2f} hours")

    # Load the dataset
    dataset = hf_datasets.load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    tokenizer.model_max_length = int(1e30)  # Suppress warnings about sequences being too long

    # Create the concat tokens dataset
    concat_dataset = ConcatTokensDataset(dataset, tokenizer, max_length=TOKENS_PER_SAMPLE)

    # Create dataloader
    loader = DataLoader(concat_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process and write samples
    file_index = 0
    sample_count = 0
    current_file = None

    progress_bar = tqdm(total=total_samples, desc="Processing samples")

    start_time = time.time()
    for batch in loader:
        for tokens in batch['tokens']:
            if sample_count % SAMPLES_PER_FILE == 0:
                if current_file:
                    current_file.close()
                file_index += 1
                file_name = f'{file_index:03d}.jsonl'
                current_file = open(os.path.join(OUTPUT_DIR, file_name), 'w')

            json_sample = {
                "token_ids": np.frombuffer(tokens, dtype=np.int64).tolist()
            }
            current_file.write(json.dumps(json_sample) + '\n')
            sample_count += 1
            progress_bar.update(1)

        if sample_count >= total_samples:
            break

        # Update time estimate every 10 iterations
        if sample_count % (BATCH_SIZE * 10) == 0:
            elapsed_time = time.time() - start_time
            iterations_completed = sample_count // BATCH_SIZE
            time_per_iteration = elapsed_time / iterations_completed
            remaining_iterations = total_iterations - iterations_completed
            estimated_remaining_time = remaining_iterations * time_per_iteration
            progress_bar.set_postfix({'Estimated remaining time': f'{estimated_remaining_time / 3600:.2f} hours'})

    if current_file:
        current_file.close()

    progress_bar.close()
    total_time = time.time() - start_time
    print(f"Processing completed. Total samples: {sample_count}")
    print(f"Total files created: {file_index}")
    print(f"Total processing time: {total_time / 3600:.2f} hours")

# Run the main function
if __name__ == "__main__":
    main()