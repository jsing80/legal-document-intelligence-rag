import logging


def get_logger(name):
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()

        # Defines the structure of each log message:
        # timestamp | severity level | module name | actual message
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Attach the formatter to the console handler
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Log INFO and all higher-severity messages
        logger.setLevel(logging.INFO)

    return logger