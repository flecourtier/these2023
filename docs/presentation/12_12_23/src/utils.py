import json
import numpy as np

def read_json_file():
    with open('config.json') as f:
        raw_config = f.read()
        config = json.loads(raw_config)
    return config