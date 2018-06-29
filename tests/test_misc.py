#! /usr/bin/env python
__author__ = 'Paul Hancock'


def test_flags():
    """Test that the flags import without errors"""
    import AegeanTools.flags
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            globals()[f]()