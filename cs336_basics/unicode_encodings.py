r"""python ./unicode_encodings.py --input_str='Hello World'"""

import cv2
from absl import app
from absl import flags

INPUT_STR = flags.DEFINE_string('input_str',
                                None,
                                'Input string for unicode encoding',
                                required=True)


def get_unicode_bytes(input_str: str):
  encoded_bytes = input_str.encode('utf-8')
  print(f"Encoded bytes: {encoded_bytes}")


def main(argv):
  del argv
  input_str = INPUT_STR.value
  get_unicode_bytes(input_str)


if __name__ == '__main__':
  app.run(main)
