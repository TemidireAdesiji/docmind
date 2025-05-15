"""Entry point for ``python -m docmind``."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv


def _silence_otel_noise() -> None:
    """Suppress noisy OpenTelemetry context-detach warnings.

    See https://github.com/open-telemetry/opentelemetry-python/issues/2606
    """

    class _OtelFilter(logging.Filter):
        def filter(self, record: logging.Record) -> bool:
            return "Failed to detach context" not in record.getMessage()

    logging.getLogger("opentelemetry.context").addFilter(_OtelFilter())


def main() -> None:
    """Bootstrap and launch the DocMind application."""
    load_dotenv(
        os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        override=False,
    )
    _silence_otel_noise()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from docmind.log import logger
    from docmind.ui.app import launch

    logger.info("Starting DocMind …")
    launch()


if __name__ == "__main__":
    main()
