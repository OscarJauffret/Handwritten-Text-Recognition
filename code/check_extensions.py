import os
from config import Config

extensions = set()

def list_extensions(dir):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isdir(path):
            list_extensions(path)
        else:
            ext = file.split('.')[-1]
            if ext != file:
                extensions.add(ext)


list_extensions(Config.Paths.data_path)
print(extensions)