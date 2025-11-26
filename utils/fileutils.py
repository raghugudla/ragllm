from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid4

import chainlit as cl


UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _persist_upload(file_bytes: bytes, filename: str) -> Path:
    """Persist the uploaded payload under data/uploads with a collision-safe name."""
    target_path = UPLOAD_DIR / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path = target_path.with_name(
            f"{target_path.stem}_{uuid4().hex[:8]}{target_path.suffix}"
        )

    with open(target_path, "wb") as out_file:
        out_file.write(file_bytes)

    return target_path


def extract_attachments(message: cl.Message) -> List[Dict[str, Any]]:
    """
    Extract file attachments from a Chainlit message, persist to disk, and return metadata.

    Returns a list of dicts with:
    - name: original filename
    - mime: MIME type (if provided by Chainlit)
    - original_path: temporary path managed by Chainlit (may be None)
    - saved_to: permanent path under data/uploads
    """
    attachments: List[Dict[str, Any]] = []
    if not getattr(message, "elements", None):
        return attachments

    for element in message.elements:
        if element.type != "file":
            continue

        if element.path:
            with open(element.path, "rb") as f:
                file_bytes = f.read()
        elif isinstance(element.content, (bytes, bytearray)):
            file_bytes = element.content
        elif isinstance(element.content, str):
            file_bytes = element.content.encode()
        else:
            file_bytes = b""

        saved_path = _persist_upload(file_bytes, element.name or "upload")

        attachments.append(
            {
                "name": element.name,
                "mime": element.mime,
                "original_path": element.path,
                "saved_to": str(saved_path),
            }
        )

    return attachments


