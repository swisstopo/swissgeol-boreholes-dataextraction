"""Contains package wide constants such as paths"""

from pathlib import Path
import os

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent
PKG_ROOT = Path(os.path.abspath(__file__)).parent
DATAPATH = PROJECT_ROOT / "data"