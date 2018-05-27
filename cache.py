import os
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


def cached():
    def decorator(fn):
        def wrapped(*args, **kwargs):
            cache_name = ""
            if fn.__name__ == "get_corpus":
                cache_name = kwargs['path'].split('/')[-1] + "_corpus"

            cache_path = "./cache/" + cache_name

            if os.path.exists(cache_path):
                print("Found the cache " + cache_path)
                with open(cache_path, 'rb') as file:
                    return pickle.load(file)

            print("Didnt find the cache " + cache_path)
            res = fn(*args, **kwargs)

            with open(cache_path, 'wb') as file:
                pickle.dump(res, file, pickle.HIGHEST_PROTOCOL)

            return res

        return wrapped

    return decorator
