"""The Byte Pair Encoding (BPE) tokenizer."""
import dataclasses
from dataclasses import field
from collections import defaultdict
import regex as re

from absl import app

# Special tokens to be put at the beginning  of the tokenizer.
_SPECIAL_TOKENS = ("<|endoftext|>",)

# REGEX pattern to pre-tokenizer the texts.
_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@dataclasses.dataclass
class BPETokenizer:
  bytes_to_idx: dict[bytes, int] = field(default_factory=dict)
  words_frequency: dict[tuple[bytes], int] = field(default_factory=dict)
  target_vocab_size: int = 1000
  current_vocab_size: int = 0

  def __post_init__(self):
    # Initialize the tokenizer with single byte tokens.
    for special_token in _SPECIAL_TOKENS:
      self.bytes_to_idx[(
          special_token.encode('utf-8'))] = self.current_vocab_size
      self.current_vocab_size += 1

    for i in range(256):
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
    new_bytes_to_original_bytes = defaultdict(list)
    for word_bytes, freq in self.words_frequency.items():
      for idx, (byte1, byte2) in enumerate(zip(word_bytes, word_bytes[1:])):
        bytes_pair = byte1 + byte2
        new_bytes_pair_freq[bytes_pair] = new_bytes_pair_freq.get(
            bytes_pair, 0) + freq
        # If this new bytes pair is merged, what would the new word bytes be?
        # [b'l', b'o', b'w'] -> [b'l', b'ow'].
        new_word_bytes = word_bytes[:idx] + (bytes_pair,) + word_bytes[idx + 2:]
        new_bytes_to_original_bytes[bytes_pair].append(
            (word_bytes, new_word_bytes))

    # Select the highest freq new bytes.
    max_freq_bytes_pair = max(new_bytes_pair_freq.items(),
                              key=lambda x: (x[1], x[0]))
    # Update the vocab.
    self.bytes_to_idx[max_freq_bytes_pair[0]] = self.vocab_size
    self.current_vocab_size += 1

    # Update the pre-tokenized words.
    for [original_bytes,
         new_bytes] in new_bytes_to_original_bytes[max_freq_bytes_pair[0]]:
      count = self.words_frequency.pop(original_bytes)
      self.words_frequency[new_bytes] = count

  @property
  def vocab_size(self):
    return self.current_vocab_size

  @property
  def vocab(self):
    return self.bytes_to_idx

  def merge_pairs_with_target_times(self, num_times: int):
    for _ in range(num_times):
      self.merge_pairs_one_time()


def main(argv):
  del argv
  input_texts = [
      "low low low low low", "lower lower widest widest widest",
      "newest newest newest newest newest newest"
  ]

  tokenizer = BPETokenizer()
  tokenizer.count_frequency(input_texts)
  tokenizer.merge_pairs_with_target_times(6)

  print("Final vocabulary: ", tokenizer.vocab)


if __name__ == '__main__':
  app.run(main)
