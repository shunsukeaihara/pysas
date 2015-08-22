# -*- coding: utf-8 -*-
from functools import wraps

def do_nothing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
