#!/usr/bin/python
# -*- coding:utf-8 -*-


"""
指定したファイルにcifar10のデータをダウンロードするスクリプト。
デフォルトは./data/
"""


import argparse
import os
import re
import sys
import tarfile
from six.moves import urllib


DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def maybe_download_and_extract(data_dir):
  """Download and extract the tarball from Alex's website."""
  dest_directory = data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", dest="dir_out", default="./data")
    args = parser.parse_args()

    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    maybe_download_and_extract(args.dir_out)


if __name__ == "__main__":
    main()
