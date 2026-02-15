"""Example script for chunking text and training the BPE tokenizer."""

import os
from typing import BinaryIO
import bpe_tokenizer
from absl import app

BASE_PATH = "/home/handeng/Desktop/Projects/CS336/homework1/data"


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
    file.seek(initial_position)  # Start at boundary gues
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


def main(argv):
  """Runs a small end-to-end pretokenization and BPE training example."""
  del argv
  dataset_name = "TinyStoriesV2-GPT4-valid.txt"
  file_path = os.path.join(BASE_PATH, dataset_name)
  tokenizer_model_name = "TinyStoriesV2-GPT4-valid_tokenizer.model"
  tokenizer_vocab_name = "TinyStoriesV2-GPT4-valid_vocab.bin"
  ## Usage

  tokenizer = bpe_tokenizer.BPETokenizer()

  with open(file_path, "rb") as f:
    num_processes = 1
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
      print(f"start: {start}, end: {end}", start, end)
      f.seek(start)
      chunk = f.read(end - start).decode("utf-8", errors="ignore")
      # Run pre-tokenization on your chunk and store the counts for each pre-token
      splitted_chunks = bpe_tokenizer.split_text_with_special_tokens(
          bpe_tokenizer.SPECIAL_TOKENS, chunk)
      print("Num of splitted chunks: ", len(splitted_chunks))
      tokenizer.train(splitted_chunks)
      break
  tokenizer_save_path = os.path.join(BASE_PATH, tokenizer_model_name)
  tokenizer_vocab_save_path = os.path.join(BASE_PATH, tokenizer_vocab_name)
  tokenizer.save(tokenizer_save_path)
  tokenizer.save_vocab(tokenizer_vocab_save_path)
  print("Full vocab: ", tokenizer.vocab)


if __name__ == '__main__':
  app.run(main)
