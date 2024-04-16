"""Contains package wide constants such as paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATAPATH = Path(os.getenv("BOREHOLES_DATA_PATH"))

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent
