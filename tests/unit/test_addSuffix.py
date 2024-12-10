#! /usr/bin/env python

import unittest
from AegeanTools.exceptions import AegeanSuffixError
from AegeanTools.CLI.aegean import addSuffix
__author__ = "Ahmed Abdelmuniem Abdalla Mohammed"


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