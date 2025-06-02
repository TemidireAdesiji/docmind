"""Langfuse-based LLM tracing.

When tracing is enabled via ``DOCMIND_TRACING_ENABLED=true``,
a Langfuse callback handler is created and attached to every
LangGraph invocation.  If credentials are missing or the
service is unreachable, tracing degrades gracefully.
"""

from __future__ import annotations

from typing import Any

from docmind.log import logger
from docmind.settings import Settings, get_settings


class TracingBridge:
    """Lifecycle wrapper around the Langfuse callback."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or get_settings()
        self._active = self._cfg.tracing_enabled
        self._handler: Any = None
        self._client: Any = None

        if not self._active:
            return

        if (
            not self._cfg.tracing_public_key
            or not self._cfg.tracing_secret_key
        ):
            logger.warning(
                "Tracing enabled but API keys are missing — disabling"
            )
            self._active = False
            return

        self._bootstrap()

    # -- public API --------------------------------------------------

    @property
    def callback(self) -> Any | None:
        """Return the Langfuse callback handler, or ``None``."""
        return self._handler

    def flush(self) -> None:
        """Flush any buffered trace data."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception:
                logger.debug("Trace flush failed (non-fatal)")

    # -- internal ----------------------------------------------------

    def _bootstrap(self) -> None:
        try:
            from langfuse import get_client
            from langfuse.langchain import CallbackHandler

            self._client = get_client()

            if self._client.auth_check():
                logger.info("Langfuse tracing authenticated")
            else:
                logger.warning(
                    "Langfuse auth check failed — disabling tracing"
                )
                self._active = False
                return

            self._handler = CallbackHandler()
        except Exception:
            logger.exception(
                "Could not initialise Langfuse — tracing disabled"
            )
            self._active = False
