#! /usr/bin/env python

from AegeanTools.CLI import aegean


def test_help():
    aegean.main()


def test_cite():
    aegean.main(['--cite'])


def test_versions():
    aegean.main(['--versions'])


def test_debug():
    aegean.main(['--debug'])
