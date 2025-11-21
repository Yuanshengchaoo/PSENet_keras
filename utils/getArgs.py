import argparse


def getArgs():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='./config/config.json',
        # default='E:\\PycharmProject\\pasi+detection\\config\\config.json',
        help='The Configuration file')
    args = argparser.parse_args()
    return args
