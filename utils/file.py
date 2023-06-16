import os
import pickle


def cleanup_paths(paths):
    for p in paths:
        os.remove(p)


def remove_filetype(filename: str):
    parts = filename.rsplit('.', maxsplit=1)
    return parts[0]
