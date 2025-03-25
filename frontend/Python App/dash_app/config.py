import json
import os

# Determina la directory base del progetto
basedir = os.path.abspath(os.path.dirname(__file__))

# Carica la configurazione dal file JSON
with open(os.path.join(basedir, "config.json"), "r") as f:
    config = json.load(f)
