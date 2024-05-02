#! /usr/bin/env python

import unittest
import os
from AegeanTools.exceptions import AegeanSuffixError

__author__ = "Ahmed Abdelmuniem Abdalla Mohammed"

def addSuffix(file, suffix):
    """
    A function to add a specified suffix before the extension.

    parameters
    ----------
    file: str
        The current name of the file
    
    suffix: str or int
        The desired suffix to be inserted before the extension
    """
    if isinstance(suffix, int):
        base, ext = os.path.splitext(file)
        base += f"_{suffix:02d}"
        fname = base + ext
    elif isinstance(suffix, str):
        if suffix[0] == "_":
            suffix = suffix[1:]
        base, ext = os.path.splitext(file)
        base += f"_{suffix}"
        fname = base + ext
    else:
        raise AegeanSuffixError(f"This suffix type is not support: {suffix}") 

    return fname

class TestAddSuffix(unittest.TestCase):

    def test_add_suffix_int(self):
        self.assertEqual(addSuffix("sample.txt", 1), "sample_01.txt")
        self.assertEqual(addSuffix("sample_2.txt", 10), "sample_2_10.txt")
        self.assertEqual(addSuffix("sample_3.txt", 100), "sample_3_100.txt")

    def test_add_suffix_str(self):
        self.assertEqual(addSuffix("sample.txt", "_comp"), "sample_comp.txt")
        self.assertEqual(addSuffix("sample_2.txt", "__comp"), "sample_2__comp.txt")

    def test_add_suffix_invalid_type(self):
        with self.assertRaises(AegeanSuffixError):
            addSuffix("sample.txt", ["invalid"])

if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith("test"):
            print(f)
            globals()[f]()