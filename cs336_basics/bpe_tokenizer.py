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
SPECIAL_TOKENS = {
    "<|endoftext|>",
}

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


def split_text_with_special_tokens_inclusive(special_tokens, s):
  """Splits text segments around special tokens including special tokens.

  Args:
    special_tokens: Iterable of literal special-token strings.
    s: Input text to split.

  Returns:
    A list of text chunks separated by any special token, with special tokens
      also within but in corresponding idx of the list following the order.
  """
  pattern = f"({'|'.join(re.escape(t) for t in special_tokens)})"
  return tuple(t for t in re.split(pattern, s) if t)


@dataclasses.dataclass
class BPETokenizer:
  """Trains and applies a byte-level Byte Pair Encoding (BPE) tokenizer.

  This dataclass stores the vocabulary state and incremental caches used
  throughout BPE training and inference.

  Attributes:
    bytes_to_idx: Mapping from token bytes to integer token IDs.
    idx_to_bytes: Mapping from integer token IDs back to token bytes.
    words_frequency: Word-frequency table where each word is a tuple of byte
      tokens.
    target_vocab_size: Desired vocabulary size after BPE merges.
    current_vocab_size: Current number of entries in the vocabulary.
    new_bytes_to_replace: Cache mapping candidate merged token pairs to words
      that contain the pair.
    new_bytes_pair_freq: Weighted frequency table for adjacent token pairs.
    merge_ranks: Mapping from merged token-byte pairs to training merge order.
  """
  bytes_to_idx: dict[bytes, int] = field(default_factory=dict)
  idx_to_bytes: dict[int, bytes] = field(default_factory=dict)
  words_frequency: dict[tuple[int, ...], int] = field(default_factory=dict)
  target_vocab_size: int = 10000
  current_vocab_size: int = 0
  special_tokens: set[str] = field(default_factory=lambda: SPECIAL_TOKENS)

  # Adjacent token pairs to set of words they should replace.
  # Should be updated each time a origin_word -> new_word
  new_bytes_to_replace: dict[tuple[int, int], set] = field(
      default_factory=lambda: defaultdict(set))

  # Adjacent token pairs to frequency counts. Should be updated each time an
  # origin_word -> new_word.
  new_bytes_pair_freq: dict[tuple[int, int], int] = field(default_factory=dict)
  merge_ranks: dict[tuple[bytes, bytes], int] = field(default_factory=dict)

  def __post_init__(self):
    """Initializes the byte-level base vocabulary and regex pretokenizer."""
    for special_token in self.special_tokens:
      self.bytes_to_idx[(
          special_token.encode('utf-8'))] = self.current_vocab_size
      self.idx_to_bytes[self.current_vocab_size] = special_token.encode('utf-8')
      self.current_vocab_size += 1

    for i in range(2**8):
      self.bytes_to_idx[bytes([i])] = self.current_vocab_size
      self.idx_to_bytes[self.current_vocab_size] = bytes([i])
      self.current_vocab_size += 1

    self.pattern = re.compile(_PATTERN)

  def count_frequency(self, input_texts: list[str]):
    """Counts pretokenized word frequencies from input texts.

    Args:
      input_texts: Raw text chunks to pretokenize and count.
    """
    for text in input_texts:
      for match in self.pattern.finditer(text):
        matched_bytes = match.group(0).encode('utf-8')
        # Convert the byte -> integer using pre-initialized vocabulary.
        word_bytes = tuple(self.bytes_to_idx[bytes([b])] for b in matched_bytes)
        self.words_frequency[word_bytes] = self.words_frequency.get(
            word_bytes, 0) + 1

  def _init_pair_cache(self):
    """Builds pair-frequency and pair-to-word caches from current word counts."""
    self.new_bytes_pair_freq.clear()
    self.new_bytes_to_replace.clear()

    for word, freq in tqdm.tqdm(self.words_frequency.items()):
      # Word is already int representation in this case.
      for b1, b2 in zip(word, word[1:]):
        pair = (b1, b2)
        self.new_bytes_pair_freq[pair] = self.new_bytes_pair_freq.get(pair,
                                                                      0) + freq
        self.new_bytes_to_replace[pair].add(word)

  def _count_adjacent_pairs(
      self, word: tuple[int, ...]) -> dict[tuple[int, int], int]:
    """Counts adjacent token-pair multiplicities within one tokenized word.

    Args:
      word: Tokenized word represented as a tuple of byte tokens.

    Returns:
      Mapping from adjacent token pair to local occurrence count.
    """
    pair_counts = {}
    for b1, b2 in zip(word, word[1:]):
      pair = (b1, b2)
      pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts

  def _merge_word_with_pair(self, word: tuple[int, ...],
                            target_pair: tuple[int, int],
                            merged_token: int) -> tuple[int, ...]:
    """Merges non-overlapping occurrences of one pair inside a word.

    Args:
      word: Original tokenized word.
      target_pair: Token pair to merge.
      merged_token: Bytes of the merged token.

    Returns:
      The word tuple after applying the non-overlapping merge rule.
    """
    merged_word = []
    i = 0
    n = len(word)
    left_token, right_token = target_pair
    while i < n:
      b1 = word[i]
      if i + 1 < n and b1 == left_token and word[i + 1] == right_token:
        merged_word.append(merged_token)
        i += 2
      else:
        merged_word.append(b1)
        i += 1
    return tuple(merged_word)

  def _remove_word_contribution(self, word: tuple[int, ...], word_freq: int,
                                pair_counts: dict[tuple[int, int], int]):
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

  def _add_word_contribution(self, word: tuple[int, ...], word_freq: int,
                             pair_counts: dict[tuple[int, int], int]):
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

  def _merge_affected_word(self, original_word: tuple[int, ...],
                           merged_pair: tuple[int, int], merged_token: int):
    """Applies one merge to a cached-affected word and updates indexes.

    Args:
      original_word: Existing word tuple that may contain ``merged_pair``.
      merged_pair: Token pair selected for this BPE merge step.
      merged_token: Int representation of the merged token.
    """
    word_freq = self.words_frequency.get(original_word)
    if word_freq is None:
      return

    old_pair_counts = self._count_adjacent_pairs(original_word)
    new_word = self._merge_word_with_pair(original_word, merged_pair,
                                          merged_token)

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
        key=lambda x:
        (x[1], self.idx_to_bytes[x[0][0]], self.idx_to_bytes[x[0][1]]),
    )
    max_freq_pair = max_freq_bytes_pair[0]
    # Bytes type.
    merged_token = self.idx_to_bytes[max_freq_pair[0]] + self.idx_to_bytes[
        max_freq_pair[1]]
    merge_pair_bytes = (self.idx_to_bytes[max_freq_pair[0]],
                        self.idx_to_bytes[max_freq_pair[1]])
    if merge_pair_bytes not in self.merge_ranks:
      self.merge_ranks[merge_pair_bytes] = len(self.merge_ranks)

    # Add merged token to vocab.
    self.bytes_to_idx[merged_token] = self.vocab_size
    self.idx_to_bytes[self.vocab_size] = merged_token
    self.current_vocab_size += 1

    affected_words = list(self.new_bytes_to_replace.get(max_freq_pair, set()))
    if not affected_words:
      # Stale entry guard.
      self.new_bytes_pair_freq.pop(max_freq_pair, None)
      self.new_bytes_to_replace.pop(max_freq_pair, None)
      return True

    for original_word in affected_words:
      self._merge_affected_word(original_word, max_freq_pair,
                                self.bytes_to_idx[merged_token])

    return True

  def save_words_freq(self, filepath):
    with open(filepath, 'wb') as f:
      pickle.dump(self.words_frequency, f)

    print(f"===== Save words frequency to {filepath} successfully. =====")

  def load_and_merge_words_freq(self, filepaths: list[str]):
    total = Counter()
    for filepath in filepaths:
      print(f"Loading word freq from {filepath}")
      with open(filepath, "rb") as f:
        total.update(pickle.load(f))
    self.words_frequency = dict(total)
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
      vocab_dict = {
          'bytes_to_idx': self.bytes_to_idx,
          'idx_to_bytes': self.idx_to_bytes,
          'vocab_size': self.current_vocab_size,
          'merge_ranks': self.merge_ranks,
      }
      pickle.dump(vocab_dict, f)
      print(f"===== Save tokenizer vocab to {filepath} successfully. =====")

  def load_vocab(self, filepath):
    """Loads the token-to-index vocabulary mapping from disk.

    Args:
      filepath: Path to a pickled vocabulary dictionary.
    """
    with open(filepath, 'rb') as f:
      vocab_dict = pickle.load(f)
      self.bytes_to_idx = vocab_dict['bytes_to_idx']
      self.idx_to_bytes = vocab_dict['idx_to_bytes']
      self.current_vocab_size = vocab_dict['vocab_size']
      self.merge_ranks = vocab_dict.get('merge_ranks', {})
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

  def _merge_encoded_word_with_pair(
      self, word: tuple[bytes, ...],
      target_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged_word = []
    left_token, right_token = target_pair
    i = 0
    while i < len(word):
      if (i + 1 < len(word) and word[i] == left_token and
          word[i + 1] == right_token):
        merged_word.append(left_token + right_token)
        i += 2
      else:
        merged_word.append(word[i])
        i += 1
    return tuple(merged_word)

  def _encode_word(self, word: list[bytes] | tuple[bytes, ...]) -> list[int]:
    new_word = tuple(word)
    while len(new_word) >= 2:
      best_pair = None
      best_rank = None
      for pair in zip(new_word, new_word[1:]):
        rank = self.merge_ranks.get(pair)
        if rank is None:
          continue
        if best_rank is None or rank < best_rank:
          best_rank = rank
          best_pair = pair

      if best_pair is None:
        break
      new_word = self._merge_encoded_word_with_pair(new_word, best_pair)

    return [self.bytes_to_idx[a] for a in new_word]

  def encode(self, text):
    chunked_texts = split_text_with_special_tokens_inclusive(
        self.special_tokens, text)
    output_tokens = []

    for text in chunked_texts:
      if text in self.special_tokens:
        output_tokens.append(self.bytes_to_idx[text.encode('utf-8')])
        continue
      word_bytes_list = []
      for match in self.pattern.finditer(text):
        matched_bytes = match.group(0).encode('utf-8')
        word_bytes_list.append(tuple(bytes([b]) for b in matched_bytes))

      for word_bytes in word_bytes_list:
        output_tokens.extend(self._encode_word(word_bytes))

    return output_tokens

  def decode(self, tokens: list[int]):
    final_output = b''
    for token in tokens:
      final_output += self.idx_to_bytes[token]

    return final_output.decode('utf-8')


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
  print(tokenizer.vocab)


if __name__ == '__main__':
  app.run(main)
