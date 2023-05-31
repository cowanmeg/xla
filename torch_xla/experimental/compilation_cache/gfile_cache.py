import os
import pathlib

class GFileCache:

  def __init__(self, path):
    """Sets up a cache at 'path'. Cached values may already be present."""
    self._path = pathlib.Path(path)
    self._path.mkdir(parents=True, exist_ok=True)

  def get(self, key):
    """Returns None if 'key' isn't present."""
    if not key:
      raise ValueError("key cannot be empty")
    path_to_key = self._path / key
    if path_to_key.exists():
      return path_to_key.read_bytes()
    else:
      return None

  def put(self, key, value):
    """Adds new cache entry."""
    if not key:
      raise ValueError("key cannot be empty")
    path_to_new_file = self._path / key
    if str(path_to_new_file).startswith('gs://'):
      # Writes to gcs are atomic.
      path_to_new_file.write_bytes(value)
    elif str(path_to_new_file).startswith('file://') or '://' not in str(path_to_new_file):
      tmp_path = self._path / f"_temp_{key}"
      with open(str(tmp_path), "wb") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
      os.rename(tmp_path, path_to_new_file)
    else:
      tmp_path = self._path / f"_temp_{key}"
      tmp_path.write_bytes(value)
      tmp_path.rename(str(path_to_new_file))