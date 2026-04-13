# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Common Kailash SDK setup patterns for MLFP exercises."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def setup_environment() -> None:
    """Load .env and validate common configuration.

    Call this at the top of every exercise that needs API keys or DB connections.
    """
    # Find .env by walking up from the exercise file
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        # Try repo root
        for parent in Path.cwd().parents:
            candidate = parent / ".env"
            if candidate.exists():
                env_path = candidate
                break

    load_dotenv(env_path)


def get_connection_manager(db_url: str | None = None):
    """Create a ConnectionManager for kailash-ml engines.

    Args:
        db_url: Database URL. Defaults to SQLite at ./mlfp.db
    """
    from kailash.db import ConnectionManager

    url = db_url or os.environ.get("DATABASE_URL", "sqlite:///mlfp.db")
    return ConnectionManager(url)


def get_device() -> "torch.device":
    """Get the best available compute device: MPS (Mac) > CUDA > CPU."""
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_llm_model() -> str:
    """Get the configured LLM model name from environment."""
    setup_environment()
    model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
    if not model:
        raise EnvironmentError(
            "No LLM model configured. Set DEFAULT_LLM_MODEL or OPENAI_PROD_MODEL in .env"
        )
    return model
