from six import reraise
from sys import exc_info


def exception_handler(func):
    """This code permits to catch SystemExit with exit code 0

    It is a decorator.
    It is for example used when "--help" is given as argument
    """
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except SystemExit as exc:
            if exc.code != 0:
                reraise(*exc_info())
    return inner_function
