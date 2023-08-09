r"""
 By Dylon Edwards

 Copyright 2023 GSI Technology, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the “Software”), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
