"""Define constants to be used throughout the repository."""
from pathlib import Path

# Main paths
PROJECT_DIR = Path(__file__).parent.parent
DEEP_GROUP_DIR = Path("/deep/group")
USER_DIR = DEEP_GROUP_DIR / "sharonz"
PROJECT_DIR = USER_DIR / "hypetrain_deep"
DATA_DIR = PROJECT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"

SAVE_MODEL_DIR = PROJECT_DIR / "saved_models"

# Dataset constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
