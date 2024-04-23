def flushed_print(func):
    def wrapped_func(*args, **kwargs):
        kwargs['flush'] = True
        return func(*args, **kwargs)
    return wrapped_func

print = flushed_print(print)