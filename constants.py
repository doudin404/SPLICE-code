import os
import sys
from dotenv import load_dotenv

IS_WINDOWS = sys.platform == 'win32'
get_trace = getattr(sys, 'gettrace', None)
DEBUG = get_trace is not None and get_trace() is not None

EPSILON = 1e-4
DIM = 3
MAX_VS = 100000
MAX_GAUSSIANS = 256

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "assets")
CHECKPOINTS_ROOT = os.path.join(DATA_ROOT, "checkpoints")
CACHE_ROOT = os.path.join(DATA_ROOT, "cache")
UI_OUT = os.path.join(DATA_ROOT, "ui_export")
UI_RESOURCES = os.path.join(DATA_ROOT, "ui_resources")
Shapenet_WT = os.path.join(DATA_ROOT, "ShapeNetCore_wt")
Shapenet = os.path.join(DATA_ROOT, "ShapeNetCore.v2")



# SPLICE
ASSETS_ROOT = os.path.join(PROJECT_ROOT, "assets")
SPLICE_CKPT_ROOT = os.path.join(ASSETS_ROOT, "checkpoints")
SPLICE_UI_OUT = os.path.join(ASSETS_ROOT, "ui_export")


load_dotenv()
DEV_MODE = os.getenv("DEV_MODE", "SPAGHETTI")
SPLICE_MODEL_PATH = os.getenv("SPLICE_MODEL_PATH", "")