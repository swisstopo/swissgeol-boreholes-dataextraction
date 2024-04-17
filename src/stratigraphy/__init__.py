"""Contains package wide constants such as paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if os.getenv("BOREHOLES_DATA_PATH") is not None:
    DATAPATH = Path(os.getenv("BOREHOLES_DATA_PATH"))
else:
    DATAPATH = Path(__file__).parent.parent.parent / "data"

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
