import logging


def setup_logger(log_path='trade.log'):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
