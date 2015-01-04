import time
from functools import wraps
from decorator import decorator

CACHE_PATH = "cache"

class Timer:
    def __enter__(self):
        print "start"
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        print "took %.4fs" % self.interval

def memo(cache, key_fun=None):
    def internalFunc(f):
        @wraps(f)
        def wrapper(f, *args, **kwargs):
            if key_fun:
                key = key_fun(*args, **kwargs)
            elif kwargs:
                key = args, frozenset(kwargs.iteritems())
            else:
                key = args
            if key in cache:
                return cache[key]
            else:
                cache[key] = result = f(*args, **kwargs)
                return result
        return decorator(wrapper, f)
    return internalFunc