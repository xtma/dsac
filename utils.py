import yaml


def load_config(path):
    try:
        with open(path, 'r', encoding="utf-8") as f:
            config = yaml.load(f)
    except IOError:
        print(f"No Such File: {path}")
    return config
