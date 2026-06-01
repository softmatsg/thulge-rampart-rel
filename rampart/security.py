"""Address randomisation for block runtime identifiers.

``generate_runtime_id`` exists so every block-creation path in the library
flows through a single source of randomness (UUID4). The property this delivers
is analogous to ASLR: an indirect prompt injection that names a specific block
by its runtime key (``"ignore block <runtime_id>"``) cannot succeed because no
stable string in the compiled context refers to that block by its key. The
human-readable ``semantic_name`` is intentionally separated from the storage
key to preserve this property even when seed file contents are public.
"""

import uuid


def generate_runtime_id() -> str:
    """Generate a fresh UUID4 string for use as a block runtime identifier.

    Wraps :func:`uuid.uuid4` so every block creation path in the library
    flows through a single source of randomness. The resulting identifier
    is unpredictable to external observers even when the seed file content
    is known, which is the property that prevents indirect prompt-injection
    attacks from naming a specific block by its runtime key.

    Returns:
        A 36-character hex-formatted UUID4 string (e.g.
        ``"550e8400-e29b-41d4-a716-446655440000"``).
    """
    return str(uuid.uuid4())
