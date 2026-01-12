import logging
from pathlib import Path
from config.paths import OUTPUTS_DIR

def setup_logging(name: str="rag_assistant"):

    log_path = Path(OUTPUTS_DIR) / f"logs/{name}.log"

    # Ensure the logs directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Line: %(lineno)d] - %(message)s')

    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.ERROR)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger