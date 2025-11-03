import logging
import os

def setup_logger(name="app_log", log_file="app.log"):
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(name)
    handler = logging.FileHandler(f"logs/{log_file}")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
