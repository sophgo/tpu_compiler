#!/usr/bin/env python3
##
## Copyright (C) Cristal Vision Technologies Inc.
## All Rights Reserved.
##

import argparse
import sys, os
from pathlib import Path
import random

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', help='dataset dir')
  parser.add_argument('--count', help='image count that want to put into list')
  parser.add_argument('--output_img_list', help='image list file')
  args = parser.parse_args()

  if (not Path(args.dataset).exists()):
    print("dataset not exist")
    sys.exit(1)

  full_list = []

  for file_path in Path(args.dataset).glob('**/*'):
    if file_path.is_file():
      full_list.append(str(file_path))
  # fixed sequance, change 4 to something else if a different list is needed
  random.Random(4).shuffle(full_list)
  # print(full_list)

  print("total {} images in dataset".format(len(full_list)))
  count = len(full_list)
  if args.count:
    count = int(args.count)
  print("select {} images into list".format(count))

  with open(args.output_img_list, 'w') as fp:
    for i in range(count):
      # print(full_list[i])
      fp.write(full_list[i])
      fp.write("\n")
