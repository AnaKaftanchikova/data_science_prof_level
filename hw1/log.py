import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='log.log', level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

def add_log_info(text_info):
    logger.info(text_info)

def add_log_warning(text_info):
    logger.warning(text_info)

def add_log_error(text_info):
    logger.exception(text_info)

def add_log_debug(text_info):
    logger.debug(text_info)