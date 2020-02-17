import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, path)

paths = list()
for folders, subfolders, files in os.walk(_ROOT):
    paths.append(os.path.join(_ROOT, files))

for path in paths:
    print(get_data(path=path))