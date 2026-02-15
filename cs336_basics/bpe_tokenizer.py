"""The Byte Pair Encoding (BPE) tokenizer."""
import dataclasses
from dataclasses import field
from collections import defaultdict
import regex as re
import tqdm
import pickle

from absl import app

# Special tokens to be put at the beginning  of the tokenizer.
SPECIAL_TOKENS = ("<|endoftext|>",)

# REGEX pattern to pre-tokenizer the texts.
_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def split_text_with_special_tokens(special_tokens, s):
  """Split a text to list of texts divided by special token."""
  return re.split(re.escape('|'.join(special_tokens)), s)


@dataclasses.dataclass
class BPETokenizer:
  bytes_to_idx: dict[bytes, int] = field(default_factory=dict)
  words_frequency: dict[tuple[bytes], int] = field(default_factory=dict)
  target_vocab_size: int = 10000
  current_vocab_size: int = 0

  def __post_init__(self):
    # Initialize the tokenizer with single byte tokens.
    for special_token in SPECIAL_TOKENS:
      self.bytes_to_idx[(
          special_token.encode('utf-8'))] = self.current_vocab_size
      self.current_vocab_size += 1

    for i in range(2**8):
      self.bytes_to_idx[bytes([i])] = self.current_vocab_size
      self.current_vocab_size += 1

    self.pattern = re.compile(_PATTERN)

  def count_frequency(self, input_texts: list[str]):
    for text in input_texts:
      for match in self.pattern.finditer(text):
        word_bytes = tuple(bytes(c, 'utf-8') for c in match.group(0))
        self.words_frequency[word_bytes] = self.words_frequency.get(
            word_bytes, 0) + 1

  def merge_pairs_one_time(self):
    new_bytes_pair_freq = {}
    # Mapping from the new byte pair to original word bytes.
    new_bytes_to_replace = defaultdict(set)
    for word_bytes, freq in self.words_frequency.items():
      for (byte1, byte2) in zip(word_bytes, word_bytes[1:]):
        bytes_pair = byte1 + byte2
        new_bytes_pair_freq[bytes_pair] = new_bytes_pair_freq.get(
            bytes_pair, 0) + freq
        new_bytes_to_replace[bytes_pair].add(word_bytes)

    if not new_bytes_pair_freq:
      return False
    max_freq_bytes_pair = max(new_bytes_pair_freq.items(),
                              key=lambda x: (x[1], x[0]))
    # Update the vocab.
    self.bytes_to_idx[max_freq_bytes_pair[0]] = self.vocab_size
    self.current_vocab_size += 1

    max_freq_bytes = max_freq_bytes_pair[0]
    # Update the pre-tokenized words.
    for original_bytes in new_bytes_to_replace[max_freq_bytes]:
      # Replace the original bytes  with new bytes
      replaced_bytes = []
      if len(original_bytes) == 1:
        continue
      i = 0
      origin_bytes_len = len(original_bytes)
      while i < origin_bytes_len:
        byte1 = original_bytes[i]
        byte2 = None
        if i < origin_bytes_len - 1:
          byte2 = original_bytes[i + 1]
        if byte2 is not None and byte1 + byte2 == max_freq_bytes:
          replaced_bytes.append(max_freq_bytes)
          i += 2
        else:
          replaced_bytes.extend([byte1])
          i += 1
      count = self.words_frequency.pop(original_bytes)
      self.words_frequency[tuple(replaced_bytes)] = count
    return True

  def save(self, filepath):
    with open(filepath, 'wb') as f:
      pickle.dump(self, f)

    print(f"===== Save tokenizer to {filepath} successfully. =====")

  def save_vocab(self, filepath):
    with open(filepath, 'wb') as f:
      pickle.dump(self.bytes_to_idx, f)
      print(f"===== Save tokenizer vocab to {filepath} successfully. =====")

  def load_vocab(self, filepath):
    with open(filepath, 'rb') as f:
      self.bytes_to_idx = pickle.load(f)
      print(f"===== Load tokenizer vocab from {filepath} successfully. =====")

  @classmethod
  def load(cls, filepath):
    with open(filepath, 'rb') as f:
      tokenizer = pickle.load(f)
      assert isinstance(tokenizer, cls)
      print(f"===== Load tokenizer from {filepath} successfully. =====")
      return tokenizer

  @property
  def vocab_size(self):
    return self.current_vocab_size

  @property
  def vocab(self):
    return self.bytes_to_idx

  def train(self, text):
    print("Counts init freq")
    self.count_frequency(text)
    for _ in tqdm.tqdm(range(self.target_vocab_size)):
      updated = self.merge_pairs_one_time()
      if self.current_vocab_size == self.target_vocab_size or not updated:
        break


def main(argv):
  del argv
  input_texts = [
      "low low low low low",
      "lower lower widest widest widest",
      "newest newest newest newest newest newest",
  ]

  tokenizer = BPETokenizer()
  tokenizer.train(input_texts)

  print("Final vocabulary: ", tokenizer.vocab)

  input_text = "low low low low low"

  print(split_text_with_special_tokens(SPECIAL_TOKENS, input_text))


if __name__ == '__main__':
  app.run(main)
