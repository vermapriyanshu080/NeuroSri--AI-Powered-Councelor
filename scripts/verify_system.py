import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_dependencies():
    """Verify all required packages are installed"""
    required_packages = [
        'numpy', 'pandas', 'torch', 'flask', 'g4f',
        'scikit-learn', 'typing_extensions'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def verify_directories():
    """Verify all required directories exist"""
    required_dirs = [
        'src/eeg', 'src/ml', 'src/chat', 'models',
        'dataset', 'logs', 'frontend'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
    
    return missing

def verify_files():
    """Verify critical files exist"""
    required_files = [
        'src/app.py',
        'src/ml/deep_emotion_model.py',
        'src/chat/chat_handler.py',
        'requirements.txt',
        'frontend/package.json'
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    return missing

def main():
    logger.info("Verifying system setup...")
    
    # Check dependencies
    missing_packages = verify_dependencies()
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Run install_dependencies.bat to install missing packages")
        return 1
        
    # Check directories
    missing_dirs = verify_directories()
    if missing_dirs:
        logger.error(f"Missing directories: {', '.join(missing_dirs)}")
        return 1
        
    # Check files
    missing_files = verify_files()
    if missing_files:
        logger.error(f"Missing files: {', '.join(missing_files)}")
        return 1
    
    logger.info("All system components verified successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 