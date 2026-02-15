"""The Byte Pair Encoding (BPE) tokenizer."""
import dataclasses
from dataclasses import field
from collections import defaultdict
import regex as re
import tqdm
from collections import Counter
import pickle

from absl import app

# Special tokens to be put at the beginning  of the tokenizer.
SPECIAL_TOKENS = ("<|endoftext|>",)

# REGEX pattern to pre-tokenizer the texts.
_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def split_text_with_special_tokens(special_tokens, s):
  """Splits text segments around special tokens.

  Args:
    special_tokens: Iterable of literal special-token strings.
    s: Input text to split.

  Returns:
    A list of text chunks separated by any special token.
  """
  return re.split(re.escape('|'.join(special_tokens)), s)


@dataclasses.dataclass
class BPETokenizer:
  """Byte Pair Encoding tokenizer trainer with incremental pair-count updates."""
  bytes_to_idx: dict[bytes, int] = field(default_factory=dict)
  words_frequency: dict[tuple[bytes], int] = field(default_factory=dict)
  target_vocab_size: int = 10000
  current_vocab_size: int = 0

  # New merged bytes to set of words it should replace.
  # Should be updated each time a origin_word -> new_word
  new_bytes_to_replace: dict[bytes, set] = field(
      default_factory=lambda: defaultdict(set))

  # Byte pairs to frequency counts. Should be updated each time an
  # origin_word -> new_word.
  new_bytes_pair_freq: dict[bytes, int] = field(default_factory=dict)

  def __post_init__(self):
    """Initializes the byte-level base vocabulary and regex pretokenizer."""
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
    """Counts pretokenized word frequencies from input texts.

    Args:
      input_texts: Raw text chunks to pretokenize and count.
    """
    for text in input_texts:
      for match in self.pattern.finditer(text):
        word_bytes = tuple(bytes(c, 'utf-8') for c in match.group(0))
        self.words_frequency[word_bytes] = self.words_frequency.get(
            word_bytes, 0) + 1

  def _init_pair_cache(self):
    """Builds pair-frequency and pair-to-word caches from current word counts."""
    self.new_bytes_pair_freq.clear()
    self.new_bytes_to_replace.clear()

    for word, freq in self.words_frequency.items():
      for b1, b2 in zip(word, word[1:]):
        pair = b1 + b2
        self.new_bytes_pair_freq[pair] = self.new_bytes_pair_freq.get(pair,
                                                                      0) + freq
        self.new_bytes_to_replace[pair].add(word)

  def _count_adjacent_pairs(self, word: tuple[bytes]) -> dict[bytes, int]:
    """Counts adjacent token-pair multiplicities within one tokenized word.

    Args:
      word: Tokenized word represented as a tuple of byte tokens.

    Returns:
      Mapping from adjacent byte-pair token to local occurrence count.
    """
    pair_counts = {}
    for b1, b2 in zip(word, word[1:]):
      pair = b1 + b2
      pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts

  def _merge_word_with_pair(self, word: tuple[bytes],
                            target_pair: bytes) -> tuple[bytes]:
    """Merges non-overlapping occurrences of one pair inside a word.

    Args:
      word: Original tokenized word.
      target_pair: Pair token to merge.

    Returns:
      The word tuple after applying the non-overlapping merge rule.
    """
    merged_word = []
    i = 0
    n = len(word)
    while i < n:
      b1 = word[i]
      if i + 1 < n and b1 + word[i + 1] == target_pair:
        merged_word.append(target_pair)
        i += 2
      else:
        merged_word.append(b1)
        i += 1
    return tuple(merged_word)

  def _remove_word_contribution(self, word: tuple[bytes], word_freq: int,
                                pair_counts: dict[bytes, int]):
    """Removes one word's weighted pair contributions from global caches.

    Args:
      word: Word whose cached contribution should be removed.
      word_freq: Corpus frequency of the word.
      pair_counts: Local adjacent-pair multiplicities for the word.
    """
    for pair, cnt in pair_counts.items():
      delta = cnt * word_freq
      prev = self.new_bytes_pair_freq.get(pair, 0)
      nxt = prev - delta
      if nxt <= 0:
        self.new_bytes_pair_freq.pop(pair, None)
      else:
        self.new_bytes_pair_freq[pair] = nxt

      words = self.new_bytes_to_replace.get(pair)
      if words is not None:
        words.discard(word)
        if not words:
          self.new_bytes_to_replace.pop(pair, None)

  def _add_word_contribution(self, word: tuple[bytes], word_freq: int,
                             pair_counts: dict[bytes, int]):
    """Adds one word's weighted pair contributions into global caches.

    Args:
      word: Word whose cached contribution should be added.
      word_freq: Corpus frequency of the word.
      pair_counts: Local adjacent-pair multiplicities for the word.
    """
    for pair, cnt in pair_counts.items():
      delta = cnt * word_freq
      self.new_bytes_pair_freq[pair] = self.new_bytes_pair_freq.get(pair,
                                                                    0) + delta
      self.new_bytes_to_replace[pair].add(word)

  def _merge_affected_word(self, original_word: tuple[bytes],
                           merged_pair: bytes):
    """Applies one merge to a cached-affected word and updates indexes.

    Args:
      original_word: Existing word tuple that may contain ``merged_pair``.
      merged_pair: Pair token selected for this BPE merge step.
    """
    word_freq = self.words_frequency.get(original_word)
    if word_freq is None:
      return

    old_pair_counts = self._count_adjacent_pairs(original_word)
    new_word = self._merge_word_with_pair(original_word, merged_pair)

    if new_word == original_word:
      return

    new_pair_counts = self._count_adjacent_pairs(new_word)
    self._remove_word_contribution(original_word, word_freq, old_pair_counts)
    self._add_word_contribution(new_word, word_freq, new_pair_counts)

    self.words_frequency.pop(original_word, None)
    self.words_frequency[new_word] = self.words_frequency.get(new_word,
                                                              0) + word_freq

  def merge_pairs_one_time(self):
    """Performs one BPE merge step using incremental cache updates.

    Returns:
      True if a merge step was executed; False if no mergeable pairs remain.
    """
    if not self.new_bytes_pair_freq:
      return False

    # Pick the most frequent pair (no heap).
    max_freq_bytes_pair = max(
        self.new_bytes_pair_freq.items(),
        key=lambda x: (x[1], x[0]),
    )
    max_freq_bytes = max_freq_bytes_pair[0]

    # Add merged token to vocab.
    self.bytes_to_idx[max_freq_bytes] = self.vocab_size
    self.current_vocab_size += 1

    affected_words = list(self.new_bytes_to_replace.get(max_freq_bytes, set()))
    if not affected_words:
      # Stale entry guard.
      self.new_bytes_pair_freq.pop(max_freq_bytes, None)
      self.new_bytes_to_replace.pop(max_freq_bytes, None)
      return True

    for original_word in affected_words:
      self._merge_affected_word(original_word, max_freq_bytes)

    return True

  def save_words_freq(self, filepath):
    with open(filepath, 'wb') as f:
      pickle.dump(self.words_frequency, f)

    print(f"===== Save words frequency to {filepath} successfully. =====")

  def load_and_merge_words_freq(self, filepaths: list[str]):
    for idx, filepath in enumerate(filepaths):
      with open(filepath, 'rb') as f:
        if idx == 0:
          self.words_frequency = pickle.load(f)
        else:
          self.words_frequency = dict(
              Counter(self.words_frequency) + Counter(pickle.load(f)))

    print("===== Load words frequency successfully. =====")

  def save(self, filepath):
    """Serializes the full tokenizer object to disk.

    Args:
      filepath: Output path for the pickled tokenizer object.
    """
    with open(filepath, 'wb') as f:
      pickle.dump(self, f)

    print(f"===== Save tokenizer to {filepath} successfully. =====")

  def save_vocab(self, filepath):
    """Serializes only the token-to-index vocabulary mapping.

    Args:
      filepath: Output path for the pickled vocabulary dictionary.
    """
    with open(filepath, 'wb') as f:
      pickle.dump(self.bytes_to_idx, f)
      print(f"===== Save tokenizer vocab to {filepath} successfully. =====")

  def load_vocab(self, filepath):
    """Loads the token-to-index vocabulary mapping from disk.

    Args:
      filepath: Path to a pickled vocabulary dictionary.
    """
    with open(filepath, 'rb') as f:
      self.bytes_to_idx = pickle.load(f)
      print(f"===== Load tokenizer vocab from {filepath} successfully. =====")

  @classmethod
  def load(cls, filepath):
    """Loads a serialized tokenizer object.

    Args:
      filepath: Path to a pickled ``BPETokenizer`` object.

    Returns:
      The deserialized tokenizer instance.
    """
    with open(filepath, 'rb') as f:
      tokenizer = pickle.load(f)
      assert isinstance(tokenizer, cls)
      print(f"===== Load tokenizer from {filepath} successfully. =====")
      return tokenizer

  @property
  def vocab_size(self):
    """Returns the current number of tokens in the vocabulary."""
    return self.current_vocab_size

  @property
  def vocab(self):
    """Returns the token-to-index vocabulary mapping."""
    return self.bytes_to_idx

  def train(self, text, init_words_freq: bool = True):
    """Trains BPE merges on the provided text chunks.

    Args:
      text: Iterable-like list of text chunks to train on.
    """
    if init_words_freq:
      self.count_frequency(text)
    print("Initialize pair cache.")
    self._init_pair_cache()
    for _ in tqdm.tqdm(range(self.target_vocab_size)):
      updated = self.merge_pairs_one_time()
      if self.current_vocab_size == self.target_vocab_size or not updated:
        break


def main(argv):
  """Runs a minimal local training example."""
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
