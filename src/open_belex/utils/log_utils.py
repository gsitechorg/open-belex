r"""
By Dylon Edwards
"""

import logging
from enum import IntEnum
from logging.handlers import RotatingFileHandler
from typing import Sequence, Type

from open_belex.utils.path_utils import user_tmp


class LogLevel(IntEnum):
    DEFAULT = logging.WARNING
    VERBOSE = logging.INFO
    DEBUG = logging.DEBUG

    @classmethod
    def names(cls: Type["LogLevel"]) -> Sequence[str]:
        names = []
        for log_level in cls:
            names.append(log_level.name)
        return names


def init_logger(logger: logging.Logger,
                script_name: str,
                log_level: int = LogLevel.DEFAULT.value,
                log_to_console: bool = True) -> None:

    logging.captureWarnings(True)

    log_dir = user_tmp() / "open-belex" / script_name
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{script_name}.log"
    log_file_exists = log_file.exists()

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(log_file, backupCount=50)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    if log_file_exists:
        file_handler.doRollover()

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)
