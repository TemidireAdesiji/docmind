"""Structured logging configuration using *loguru*."""

from __future__ import annotations

import sys

from loguru import logger

# Remove the default stderr sink so we can reconfigure it.
logger.remove()

logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
        "<cyan>{line}</cyan> — <level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)

__all__ = ["logger"]
