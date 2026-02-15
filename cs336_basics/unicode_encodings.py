r"""python ./unicode_encodings.py --input_str='Hello World'"""

import cv2
from absl import app
from absl import flags
import regex as re

_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

INPUT_STR = flags.DEFINE_string('input_str', "Random texts",
                                'Input string for unicode encoding')


def get_unicode_bytes(input_str: str):
  encoded_bytes = input_str.encode('utf-8')
  print(f"Encoded bytes: {encoded_bytes}")
  for b in encoded_bytes:

    print(f"Byte value: {bytes([b])}")
  encoded_bytes_list = list(encoded_bytes)
  print(f"Encoded bytes as list of integers: {encoded_bytes_list}")


def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
  return "".join(
      [bytes([b]).decode("utf-8", errors='replace') for b in bytestring])


def main(argv):
  del argv
  # input_str = INPUT_STR.value
  # #get_unicode_bytes(input_str)
  # result = decode_utf8_bytes_to_str_wrong('牛'.encode('utf-8'))
  # print(f"Decoded result: {result}")
  print(re.findall(_PATTERN, "<|endoftext|>"))

  for match in re.finditer(_PATTERN, "some text that i'll pre-tokenize"):
    print(f"Match: '{match.group(0)}' at position {match.span()}")


if __name__ == '__main__':
  app.run(main)
