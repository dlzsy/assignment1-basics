"""Script for multi-process chunked text and training the BPE tokenizer."""

import os
from typing import BinaryIO
import bpe_tokenizer
from absl import app, flags
from multiprocessing import Pool
from pathlib import Path
import re

BASE_PATH = "/home/handeng/Desktop/Projects/CS336/homework1/data"

_NUM_PROCESSES = flags.DEFINE_integer('num_processes', 28,
                                      "Num of parallel jobs to pre-tokenize.")

_TARGET_VOCAB_SIZE = flags.DEFINE_integer('target_vocab_size', 10000,
                                          "Target vocabulary size.")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
  """Finds chunk boundaries aligned to a special-token byte sequence.

  The returned boundaries can be used to split a file into independently
  processable segments while avoiding cuts through ``split_special_token``.

  Args:
    file: Open binary file object positioned anywhere.
    desired_num_chunks: Target number of chunks.
    split_special_token: Special-token bytes used as valid split points.

  Returns:
    Sorted unique boundary offsets including ``0`` and ``file_size``.
  """
  assert isinstance(split_special_token,
                    bytes), "Must represent special token as a bytestring"

  # Get total file size in bytes
  file.seek(0, os.SEEK_END)
  file_size = file.tell()
  file.seek(0)

  chunk_size = file_size // desired_num_chunks

  # Initial guesses for chunk boundary locations, uniformly spaced
  # Chunks start on previous index, don't include last index
  chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
  chunk_boundaries[-1] = file_size

  mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

  for bi in range(1, len(chunk_boundaries) - 1):
    initial_position = chunk_boundaries[bi]
    file.seek(initial_position)  # Start at boundary guess
    while True:
      mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

      # If EOF, this boundary should be at the end of the file
      if mini_chunk == b"":
        chunk_boundaries[bi] = file_size
        break

      # Find the special token in the mini chunk
      found_at = mini_chunk.find(split_special_token)
      if found_at != -1:
        chunk_boundaries[bi] = initial_position + found_at
        break
      initial_position += mini_chunk_size

  # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
  return sorted(set(chunk_boundaries))


def split_doc_to_chunks(args):
  file_path, boundary_start, boundary_end, process_idx = args
  with open(file_path, "rb") as f:
    f.seek(boundary_start)
    chunk = f.read(boundary_end - boundary_start).decode("utf-8",
                                                         errors="ignore")
  # Run pre-tokenization on your chunk and store the counts for each pre-token
  splitted_chunks = bpe_tokenizer.split_text_with_special_tokens(
      bpe_tokenizer.SPECIAL_TOKENS, chunk)
  print(
      f"Num of splitted chunks {len(splitted_chunks)} for process {process_idx}"
  )
  tokenizer = bpe_tokenizer.BPETokenizer(
      target_vocab_size=_TARGET_VOCAB_SIZE.value)
  # Counts the initial frequency -> Get words_frequency
  tokenizer.count_frequency(splitted_chunks)
  file_path_obj = Path(file_path)
  parent_dir = file_path_obj.parent
  stem = file_path_obj.stem
  tokenizer.save_words_freq(
      os.path.join(parent_dir, 'vocab_freq', f'{stem}_{str(process_idx)}.bin'))


def train_tokenizer(split='train', dataset_name='TinyStoriesV2-GPT4-'):
  dataset_name = f"{dataset_name}{split}.txt"
  file_path = os.path.join(BASE_PATH, dataset_name)
  num_processes = _NUM_PROCESSES.value
  tokenizer_model_name = f"{dataset_name}{split}_tokenizer.model"
  tokenizer_vocab_name = f"{dataset_name}{split}_vocab.bin"

  words_freq_base_dir = os.path.join(BASE_PATH, 'vocab_freq')
  if not os.path.exists(words_freq_base_dir):
    os.makedirs(words_freq_base_dir)

  arg_list = []
  with open(file_path, "rb") as f:
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

  arg_list.extend([file_path, boundaries[i], boundaries[i + 1], i]
                  for i in range(num_processes))

  with Pool(processes=num_processes) as pool:
    _ = pool.map(split_doc_to_chunks, arg_list)

  file_path_obj = Path(file_path)
  parent_dir = file_path_obj.parent
  stem = file_path_obj.stem
  all_words_freq_path = [
      os.path.join(parent_dir, 'vocab_freq', f'{stem}_{str(process_idx)}.bin')
      for process_idx in range(num_processes)
  ]
  tokenizer = bpe_tokenizer.BPETokenizer(
      target_vocab_size=_TARGET_VOCAB_SIZE.value)
  # Load all pre-tokenized words frequency.
  tokenizer.load_and_merge_words_freq(all_words_freq_path)
  tokenizer.train(text=None, init_words_freq=False)
  tokenizer.save(os.path.join(BASE_PATH, tokenizer_model_name))
  tokenizer.save(os.path.join(BASE_PATH, tokenizer_vocab_name))


def main(argv):
  """Runs a small end-to-end pretokenization and BPE training example."""
  del argv
  dataset_name = 'TinyStoriesV2-GPT4-'
  split = 'train'

  tokenizer_vocab_name = f"{dataset_name}{split}_vocab.bin"

  # Train the tokenizer.
  # train_tokenizer(split=split, dataset_name=dataset_name)

  input_text = "A random text to be tokenized. <|endoftext|> What is your name?"

  tokenizer = bpe_tokenizer.BPETokenizer.load(
      os.path.join(BASE_PATH, tokenizer_vocab_name))
  tokens = tokenizer.encode(input_text)
  print("Input text to be tokenize: ", input_text)
  print("Tokenized: : ", tokens)
  detokenized = tokenizer.decode(tokens)
  print("De-tokenized:", detokenized)
  assert detokenized == input_text


if __name__ == '__main__':
  app.run(main)
