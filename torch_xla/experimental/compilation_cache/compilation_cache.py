import pathlib
from torch_xla.experimental.compilation_cache.gfile_cache import GFileCache

# Global cache object
_cache = None

def initialize_cache(path):
    print("Initialize persistent compilation cache")
    global _cache
    # if _cache is not None and _cache._path == pathlib.Path(path):
    #     logger.warning("Cache already previously initialized at %s", _cache._path)
    #     return

    # assert (
    #     _cache is None
    # ), f"The cache path has already been initialized to {_cache._path}"
    # _cache = GFileCache(path)
    # logger.warning("Initialized persistent compilation cache at %s", path)
    if _cache is not None and _cache._path == pathlib.Path(path):
        print(f"Cache already previously initialized at {_cache._path}")
        return
    _cache = GFileCache(path)

def is_initialized():
    return _cache is not None

def reset_cache():
    print("Deleting programs in cache")

def get_executable(cache_key, compilation_options, backend):
    print(f"Checking if a cached program is in the {backend} cache")


def put_executable(cache_key, module_name, executable, backend):
    print(f"Adding {module_name} to {backend} compilation cache")


def get_cache_key(module, devices, compile_options, backend):
    print(f"Generating cache key for {backend}")