#! python
__author__ = 'Paul Hancock'
__date__ = ''


def test_flags():
    import AegeanTools.flags
    return


if __name__ == "__main__":
    # introspect and run all the functions starting with 'test'
    for f in dir():
        if f.startswith('test'):
            print(f)
            exec(f+"()")