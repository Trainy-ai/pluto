"""Stub for wandb.util — utility helpers.

Provides the handful of wandb.util functions that user code sometimes
calls directly.
"""

import random
import string


def generate_id(length=8):
    """Generate a random run id (same format as wandb)."""
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choices(chars, k=length))


def make_artifact_name_safe(name):
    """Sanitize artifact name."""
    return name.replace('/', '-').replace('\\', '-')


def to_json(obj):
    """Best-effort JSON conversion."""
    import json

    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return str(obj)
