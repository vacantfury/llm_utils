"""Logger shim — previously the host repos' ``src.utils.logger``.

Library-clean: returns a stdlib logger and configures nothing; consumers own logging
configuration.
"""
import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
