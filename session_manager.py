import atexit
import os
import shutil
import tempfile
import uuid
import weakref
from pathlib import Path

BASE_TEMP_DIR = Path(tempfile.gettempdir()) / "qna_bot_sessions"


@atexit.register
def cleanup_all_sessions() -> None:
    """Wipe all session temp folders when the server shuts down."""
    shutil.rmtree(BASE_TEMP_DIR, ignore_errors=True)


class SessionFileManager:
    """Manages isolated, per-session file storage with automatic lifecycle cleanup."""

    def __init__(self) -> None:
        self.session_id: str = str(uuid.uuid4())
        self.session_dir: Path = BASE_TEMP_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.current_file_path: str | None = None

        # Automatically delete the session directory when this object is garbage-collected on session expiry
        self._finalizer = weakref.finalize(
            self, shutil.rmtree, str(self.session_dir), ignore_errors=True
        )

    def save_file(self, file_name: str, file_bytes: bytes) -> str:
        """Save an uploaded file exclusively within this session's sandbox."""
        self.cleanup_current_file()

        # Sanitize filename to avoid directory traversal
        safe_name = Path(file_name).name
        target_path = self.session_dir / safe_name
        with open(target_path, "wb") as f:
            f.write(file_bytes)

        self.current_file_path = str(target_path)
        return self.current_file_path

    def cleanup_current_file(self) -> None:
        """Remove the active file for this session if it exists."""
        if self.current_file_path and os.path.exists(self.current_file_path):
            try:
                os.remove(self.current_file_path)
            except OSError:
                pass
            self.current_file_path = None

    def get_file_bytes(self) -> bytes | None:
        """Read the session's active file if needed."""
        if self.current_file_path and os.path.exists(self.current_file_path):
            with open(self.current_file_path, "rb") as f:
                return f.read()
        return None
