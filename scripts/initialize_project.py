from pathlib import Path
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.ml.model_init import initialize_model

def main():
    try:
        # Create necessary directories
        (project_root / "models").mkdir(exist_ok=True)
        (project_root / "dataset").mkdir(exist_ok=True)
        (project_root / "logs").mkdir(exist_ok=True)
        
        # Initialize model
        model_path = initialize_model()
        logger.info(f"Model initialized at {model_path}")
        
        return 0
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 