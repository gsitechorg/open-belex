r"""
By Dylon Edwards
"""

import logging
import logging.handlers
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import yaml

from cerberus import Validator

from open_belex.common.constants import (MAX_L1_VALUE, MAX_L2_VALUE,
                                         MAX_RN_VALUE)
from open_belex.common.register_arenas import (NUM_EWE_REGS, NUM_L1_REGS,
                                               NUM_L2_REGS, NUM_RE_REGS,
                                               NUM_RN_REGS, NUM_SM_REGS)
from open_belex.common.stack_manager import StackManager

LOGGER = logging.getLogger()

CONFIG = "Config"

CONFIG_SCHEMA = {
    "max_rn_regs": {
        "type": "integer",
        "min": 1,
        "max": NUM_RN_REGS,
    },
    "reservations": {
        "type": "dict",
        "schema": {
            "sm_regs": {
                "type": "list",
                "schema": {
                    "type": "integer",
                    "min": 0,
                    "max": NUM_SM_REGS - 1,
                },
            },
            "rn_regs": {
                "type": "list",
                "schema": {
                    "type": "integer",
                    "min": 0,
                    "max": NUM_RN_REGS - 1,
                },
            },
            "re_regs": {
                "type": "list",
                "schema": {
                    "type": "integer",
                    "min": 0,
                    "max": NUM_RE_REGS - 1,
                },
            },
            "ewe_regs": {
                "type": "list",
                "schema": {
                    "type": "integer",
                    "min": 0,
                    "max": NUM_EWE_REGS - 1,
                },
            },
            "l1_regs": {
                "type": "list",
                "schema": {
                    "type": "integer",
                    "min": 0,
                    "max": NUM_L1_REGS - 1,
                },
            },
            "l2_regs": {
                "type": "list",
                "schema": {
                    "type": "integer",
                    "min": 0,
                    "max": NUM_L2_REGS - 1,
                },
            },
            "row_numbers": {
                "type": "list",
                "schema": {
                    "type": "integer",
                    "min": 0,
                    "max": MAX_RN_VALUE,
                },
            },
            "l1_rows": {
                "type": "list",
                "schema": {
                    "type": "integer",
                    "min": 0,
                    "max": MAX_L1_VALUE,
                },
            },
            "l2_rows": {
                "type": "list",
                "schema": {
                    "type": "integer",
                    "min": 0,
                    "max": MAX_L2_VALUE,
                },
            },
        },
    },
}


def validate_config(config: Dict[str, Any]) -> None:
    config_validator = Validator(CONFIG_SCHEMA)
    if not config_validator.validate(config):
        error_message = \
            f"Validation failed for config: {config_validator.errors}"
        raise RuntimeError(error_message)


def load_config(config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    global CONFIG_SCHEMA

    if config_path is None:
        return {}

    if not isinstance(config_path, Path):
        config_path = Path(config_path)

    if not config_path.exists():
        raise ValueError(f"File not found: {config_path}")

    with open(config_path, "rt") as f:
        config = yaml.safe_load(f)

    try:
        validate_config(config)
    except RuntimeError as error:
        error_message = \
            f"Validation failed for {config_path}"
        raise RuntimeError(error_message) from error

    return config


def belex_config(**config) -> Callable:
    validate_config(config)

    def decorator(fn: Callable) -> Callable:

        @wraps(fn)
        def wrapper(*args, **kwargs):
            StackManager.push(CONFIG, config)
            retval = fn(*args, **kwargs)
            popped = StackManager.pop(CONFIG)
            assert config is popped
            return retval

        for attribute, value in fn.__dict__.items():
            setattr(wrapper, attribute, value)

        return wrapper

    return decorator
