"""
Custom tools available to Victor's LangGraph agent.

Each tool is a plain Python function decorated with ``@tool`` from
LangChain.  New capabilities (e.g. smart-home control, calendar, database
reads) can be added here without modifying the agent graph itself.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from langchain_core.tools import tool  # type: ignore[import-untyped]

    _LC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LC_AVAILABLE = False

    def tool(fn: Any) -> Any:  # type: ignore[misc]
        """No-op decorator when langchain_core is unavailable."""
        return fn


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


if _LC_AVAILABLE:
    from langchain_core.tools import tool as _tool_decorator
else:
    _tool_decorator = tool  # type: ignore[assignment]


@_tool_decorator
def read_local_file(file_path: str) -> str:
    """Read and return the contents of a local file.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the file to read.

    Returns
    -------
    str
        The file contents as a UTF-8 string, or an error message.
    """
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            return f"Error: file not found: {file_path}"
        if not path.is_file():
            return f"Error: path is not a file: {file_path}"
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("read_local_file failed: %s", exc)
        return f"Error reading file: {exc}"


@_tool_decorator
def list_directory(directory_path: str) -> str:
    """List files and subdirectories inside *directory_path*.

    Parameters
    ----------
    directory_path:
        The directory to list.

    Returns
    -------
    str
        A newline-separated list of names, or an error message.
    """
    try:
        path = Path(directory_path).resolve()
        if not path.exists():
            return f"Error: directory not found: {directory_path}"
        if not path.is_dir():
            return f"Error: path is not a directory: {directory_path}"
        entries = sorted(p.name for p in path.iterdir())
        return "\n".join(entries) if entries else "(empty directory)"
    except OSError as exc:
        return f"Error listing directory: {exc}"


@_tool_decorator
def fetch_url(url: str) -> str:
    """Fetch the content of a URL and return it as a UTF-8 string.

    Parameters
    ----------
    url:
        The HTTP/HTTPS URL to retrieve.

    Returns
    -------
    str
        The response body, or an error message.
    """
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("fetch_url failed: %s", exc)
        return f"Error fetching URL: {exc}"


@_tool_decorator
def send_email(to_address: str, subject: str, body: str) -> str:
    """Send a plain-text email via SMTP.

    Reads SMTP configuration from environment variables:
    ``SMTP_HOST``, ``SMTP_PORT``, ``SMTP_USER``, ``SMTP_PASSWORD``,
    ``SMTP_FROM``.

    Parameters
    ----------
    to_address:
        Recipient email address.
    subject:
        Email subject line.
    body:
        Plain-text email body.

    Returns
    -------
    str
        ``"sent"`` on success or an error message.
    """
    host = os.getenv("SMTP_HOST", "localhost")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    from_addr = os.getenv("SMTP_FROM", user)

    try:
        msg = EmailMessage()
        msg["From"] = from_addr
        msg["To"] = to_address
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP(host, port, timeout=10) as smtp:
            smtp.starttls()
            if user:
                smtp.login(user, password)
            smtp.send_message(msg)
        return "sent"
    except Exception as exc:
        logger.warning("send_email failed: %s", exc)
        return f"Error sending email: {exc}"


@_tool_decorator
def parse_json(json_string: str) -> str:
    """Parse a JSON string and return a pretty-printed version.

    Parameters
    ----------
    json_string:
        A valid JSON string.

    Returns
    -------
    str
        Indented JSON string or an error message.
    """
    try:
        data = json.loads(json_string)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as exc:
        return f"Error parsing JSON: {exc}"


# Exported list for easy registration with the agent executor.
ALL_TOOLS = [
    read_local_file,
    list_directory,
    fetch_url,
    send_email,
    parse_json,
]
