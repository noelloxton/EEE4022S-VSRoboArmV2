#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os

"""
path = "~/arm_testing_ws/src/panda_simulation/data_for_nn/label.txt"
def main():
    with open("label.txt", "r") as f:
    	words = f.read().splitlines()
    	f.close()
    print(words)
"""

def main():
    mylist = sorted(os.listdir('/home/ghostlini/arm_testing_ws/src/panda_simulation/data_for_nn'))
    mylist.remove('label.txt')
    print(mylist)


if __name__== "__main__":
    main()
