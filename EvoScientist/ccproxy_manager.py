"""ccproxy lifecycle management for OAuth-based Anthropic access.

Provides functions to start/stop/health-check ccproxy, which allows
users with a Claude Pro/Max subscription to use EvoScientist without
a separate API key by reusing Claude Code's OAuth tokens.

ccproxy is invoked via subprocess (not Python imports) so the
``ccproxy-api`` package is truly optional at runtime.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

from EvoScientist.config import EvoScientistConfig

logger = logging.getLogger(__name__)

_AUTH_STATUS_TIMEOUT_SECONDS = 30


# =============================================================================
# Availability & auth checks
# =============================================================================


def _ccproxy_exe() -> str | None:
    """Return the path to the ccproxy binary, or None if not found.

    Checks PATH first, then the current Python environment's bin directory
    (handles conda envs where newly installed binaries may not be visible
    to shutil.which immediately after pip install).
    """
    found = shutil.which("ccproxy")
    if found:
        return found
    import sys as _sys

    candidate = os.path.join(os.path.dirname(_sys.executable), "ccproxy")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def is_ccproxy_available() -> bool:
    """Check whether the ``ccproxy`` CLI binary is available."""
    return _ccproxy_exe() is not None


def _is_editable_install() -> bool:
    """Return True if EvoScientist was installed in editable/development mode.

    Checks all matching distributions because a stale ``.egg-info`` in the
    project root can shadow the real ``dist-info`` in site-packages.
    """
    try:
        import importlib.metadata as _meta
        import json

        for dist in _meta.distributions():
            name = dist.metadata.get("Name", "")
            if name.lower() != "evoscientist":
                continue
            direct_url = dist.read_text("direct_url.json")
            if direct_url is not None:
                data = json.loads(direct_url)
                if data.get("dir_info", {}).get("editable", False) is True:
                    return True
    except Exception:
        pass
    return False


def _oauth_install_hint() -> str:
    """Return the appropriate install command depending on install method."""
    if _is_editable_install():
        return "uv sync --extra oauth or pip install -e '.[oauth]'"
    return "pip install 'evoscientist[oauth]'"


def _summarize_auth_output(raw: str) -> str:
    """Extract key fields from ccproxy auth status output into a one-line summary.

    Parses the Rich table output for Email, Subscription, and Status fields.
    Returns e.g. ``"user@example.com (plus, active)"``.
    Falls back to ``"Authenticated"`` if parsing fails.
    """
    import re as _re

    # Strip ANSI escape sequences
    clean = _re.sub(r"\x1b\[[0-9;]*m", "", raw)

    # Parse "Key<2+ spaces>Value" table rows, match exact key names
    fields: dict[str, str] = {}
    for line in clean.splitlines():
        m = _re.match(r"\s*(.+?)\s{2,}(.+)", line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        if key in ("Email", "Subscription", "Subscription Status"):
            fields[key.lower().replace(" ", "_")] = val

    email = fields.get("email", "")
    sub = fields.get("subscription", "")
    status = fields.get("subscription_status", "")

    if email:
        detail = ", ".join(filter(None, [sub, status]))
        return f"{email} ({detail})" if detail else email
    return "Authenticated"


def check_ccproxy_auth(provider: str = "claude_api") -> tuple[bool, str]:
    """Check if ccproxy has valid OAuth credentials.

    Args:
        provider: ccproxy provider name ("claude_api" or "codex").

    Returns:
        (is_valid, message) tuple.
    """
    try:
        exe = _ccproxy_exe() or "ccproxy"
        result = subprocess.run(
            [exe, "auth", "status", provider],
            capture_output=True,
            text=True,
            timeout=_AUTH_STATUS_TIMEOUT_SECONDS,
        )
        import re as _re

        raw = (result.stdout + result.stderr).strip()
        clean = _re.sub(r"\x1b\[[0-9;]*m", "", raw)

        # Filter out structlog warning/noise lines, keep only status lines
        status_lines = [
            line
            for line in clean.splitlines()
            if line.strip()
            and not _re.match(r"\d{4}-\d{2}-\d{2}", line.strip())
            and "warning" not in line.lower()
            and "plugin" not in line.lower()
        ]
        status_msg = " ".join(status_lines).strip()

        # ccproxy auth status may exit 0 even when not authenticated —
        # detect failure by checking output content
        if result.returncode != 0 or "not authenticated" in clean.lower():
            return False, status_msg or "Not authenticated"

        summary = _summarize_auth_output(result.stdout)
        return True, summary or "Authenticated"
    except FileNotFoundError:
        return False, "ccproxy not found"
    except subprocess.TimeoutExpired:
        return False, "Auth check timed out"
    except Exception as exc:
        return False, f"Auth check failed: {exc}"


# =============================================================================
# Process management
# =============================================================================


def is_ccproxy_running(port: int) -> bool:
    """Check if ccproxy is already serving on the given port."""
    import httpx

    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/health/live", timeout=2.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, OSError):
        return False


def _build_ccproxy_env() -> dict[str, str]:
    """Build subprocess env for ccproxy with explicit proxy propagation.

    ccproxy may be launched from shells that do not export proxy vars
    consistently. Mirror both upper/lower-case names and default to the
    user's local Clash-style proxy on 127.0.0.1:7890 when none is set.
    """
    env = os.environ.copy()
    proxy_url = (
        env.get("ALL_PROXY")
        or env.get("all_proxy")
        or env.get("HTTPS_PROXY")
        or env.get("https_proxy")
        or env.get("HTTP_PROXY")
        or env.get("http_proxy")
        or "http://127.0.0.1:7890"
    )

    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env[key] = proxy_url

    no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or ""
    no_proxy_items = [item.strip() for item in no_proxy.split(",") if item.strip()]
    for host in ("localhost", "127.0.0.1", "::1"):
        if host not in no_proxy_items:
            no_proxy_items.append(host)
    merged_no_proxy = ",".join(no_proxy_items)
    env["NO_PROXY"] = merged_no_proxy
    env["no_proxy"] = merged_no_proxy
    return env


def _build_ccproxy_command(
    port: int, *, anthropic_oauth: bool = False, openai_oauth: bool = False
) -> list[str]:
    """Build the ccproxy serve command for the requested provider mix."""
    exe = _ccproxy_exe() or "ccproxy"
    cmd = [exe, "serve", "--port", str(port)]

    # OpenAI-only sessions do not need Claude plugin bootstrapping.
    if openai_oauth and not anthropic_oauth:
        for plugin_name in ("claude_api", "oauth_claude", "claude_sdk"):
            cmd.extend(["--disable-plugin", plugin_name])

    # Anthropic-only sessions do not need Codex plugin bootstrapping.
    if anthropic_oauth and not openai_oauth:
        for plugin_name in ("codex", "oauth_codex", "copilot"):
            cmd.extend(["--disable-plugin", plugin_name])

    return cmd


def start_ccproxy(
    port: int, *, anthropic_oauth: bool = False, openai_oauth: bool = False
) -> subprocess.Popen:
    """Start ccproxy serve as a background process.

    Args:
        port: Port number for the proxy server.

    Returns:
        The Popen handle for the ccproxy process.

    Raises:
        RuntimeError: If ccproxy fails to become healthy within 30 seconds.
        FileNotFoundError: If ccproxy binary is not found.
    """
    cmd = _build_ccproxy_command(
        port, anthropic_oauth=anthropic_oauth, openai_oauth=openai_oauth
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_build_ccproxy_env(),
    )

    # Wait for health (ccproxy can take up to ~11s on first start)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"ccproxy exited immediately with code {proc.returncode}"
            )
        if is_ccproxy_running(port):
            return proc
        time.sleep(0.3)

    # Timed out — clean up
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
    raise RuntimeError("ccproxy did not become healthy within 30 seconds")


def stop_ccproxy(proc: subprocess.Popen | None) -> None:
    """Gracefully stop a ccproxy process.

    Safe to call with None (no-op).
    """
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    except Exception:
        pass


def ensure_ccproxy(
    port: int, *, anthropic_oauth: bool = False, openai_oauth: bool = False
) -> subprocess.Popen | None:
    """Ensure ccproxy is running — reuse existing or start new.

    Returns:
        Popen handle if we started a new process, None if already running.
    """
    if is_ccproxy_running(port):
        logger.debug("ccproxy already running on port %d", port)
        return None
    return start_ccproxy(
        port, anthropic_oauth=anthropic_oauth, openai_oauth=openai_oauth
    )


# =============================================================================
# Environment setup
# =============================================================================


def setup_ccproxy_env(port: int) -> None:
    """Set environment variables for Anthropic ccproxy routing.

    Force-sets ``ANTHROPIC_BASE_URL`` and ``ANTHROPIC_API_KEY`` so that
    downstream LangChain/Anthropic clients route through ccproxy.

    Always overrides existing values — when this function is called,
    we've decided to use ccproxy, so env must point to it.
    """
    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}/claude"
    os.environ["ANTHROPIC_API_KEY"] = "ccproxy-oauth"


def setup_codex_env(port: int) -> None:
    """Set environment variables for OpenAI/Codex ccproxy routing.

    Force-sets ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` so that
    downstream LangChain/OpenAI clients route through ccproxy's Codex
    endpoint.

    Always overrides existing values — when this function is called,
    we've decided to use ccproxy, so env must point to it.
    """
    os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/codex/v1"
    os.environ["OPENAI_API_KEY"] = "ccproxy-oauth"


# =============================================================================
# High-level orchestration
# =============================================================================


def _patch_ccproxy_oauth_header() -> None:
    """Auto-patch ccproxy's adapter to send the correct OAuth beta header.

    ccproxy 0.2.4 hardcodes ``computer-use-2025-01-24`` as the
    ``anthropic-beta`` header, which causes two problems:
    - Missing ``oauth-2025-04-20`` → 401 from Anthropic
    - ``computer-use-2025-01-24`` incompatible with OAuth auth → 400

    The ccproxy binary may use a different Python environment than the one
    running EvoScientist, so we resolve the adapter path via the ccproxy
    binary's shebang line rather than the current Python's import system.

    This patch is idempotent and places the header AFTER cli_headers so it
    cannot be overridden.
    """
    import pathlib
    import re
    import sys

    try:
        ccproxy_bin = _ccproxy_exe()
        if not ccproxy_bin:
            return

        ccproxy_path = pathlib.Path(ccproxy_bin)
        if ccproxy_path.suffix.lower() == ".exe":
            # Windows console_scripts use an .exe launcher, not a text shebang.
            candidate = ccproxy_path.resolve().parent.parent / "python.exe"
            python_exe = str(candidate) if candidate.exists() else sys.executable
        else:
            # POSIX console_scripts are regular text entrypoints with a shebang.
            lines = ccproxy_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if not lines or not lines[0].startswith("#!"):
                python_exe = sys.executable
            else:
                python_exe = lines[0].lstrip("#!").strip()

        # Ask that Python where ccproxy's adapter lives
        result = subprocess.run(
            [
                python_exe,
                "-c",
                "import inspect, ccproxy.plugins.claude_api.adapter as m; print(inspect.getfile(m))",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return
        src_file = pathlib.Path(result.stdout.strip())
        if not src_file.exists():
            return

        text = src_file.read_text(encoding="utf-8")

        # Check if already correctly patched (oauth header set after cli_headers)
        correct = 'filtered_headers["anthropic-beta"] = "oauth-2025-04-20"'
        cli_marker = "cli_headers = self._collect_cli_headers()"
        if correct in text:
            # Verify it's placed after cli_headers
            if text.index(correct) > text.index(cli_marker):
                return  # Already correctly patched

        # Move/replace anthropic-beta assignment to after cli_headers loop,
        # and set only oauth-2025-04-20 (computer-use-* is incompatible).
        # Step 1: remove any existing filtered_headers["anthropic-beta"] line
        patched = re.sub(
            r'\s*filtered_headers\["anthropic-beta"\]\s*=\s*"[^"]*"\n',
            "\n",
            text,
        )
        # Step 2: insert correct assignment after the cli_headers block
        insert_after = "filtered_headers[lk] = value\n"
        replacement = (
            "filtered_headers[lk] = value\n\n"
            "        # oauth-2025-04-20: required for OAuth Bearer token auth (Anthropic 2026-03)\n"
            '        filtered_headers["anthropic-beta"] = "oauth-2025-04-20"\n'
        )
        patched = patched.replace(insert_after, replacement, 1)

        if patched == text:
            return

        src_file.write_text(patched, encoding="utf-8")
        for pyc in src_file.parent.glob("__pycache__/adapter*.pyc"):
            pyc.unlink(missing_ok=True)
        logger.info("Auto-patched ccproxy adapter: set anthropic-beta=oauth-2025-04-20")
    except Exception as exc:
        logger.warning("Could not auto-patch ccproxy adapter: %s", exc)


def maybe_start_ccproxy(config: EvoScientistConfig) -> subprocess.Popen | None:
    """High-level: conditionally start ccproxy based on config.

    Checks ``config.anthropic_auth_mode`` and ``config.openai_auth_mode``:
    - ``oauth``: ccproxy must work — raises on failure.
    - ``api_key``: no-op for that provider.

    When either provider uses OAuth, ccproxy is started (single process
    serves both providers). Environment variables are set for each
    provider that uses OAuth.

    Args:
        config: An ``EvoScientistConfig`` instance.

    Returns:
        Popen handle if we started ccproxy, None otherwise.
    """
    anthropic_oauth = getattr(config, "anthropic_auth_mode", "api_key") == "oauth"
    openai_oauth = getattr(config, "openai_auth_mode", "api_key") == "oauth"

    if not anthropic_oauth and not openai_oauth:
        return None

    if not is_ccproxy_available():
        raise RuntimeError(
            "ccproxy is required for OAuth mode but not found. "
            f"Install it with: {_oauth_install_hint()}"
        )

    # Check auth for each provider that uses OAuth
    if anthropic_oauth:
        authed, msg = check_ccproxy_auth("claude_api")
        if not authed:
            raise RuntimeError(
                f"ccproxy Anthropic OAuth not authenticated: {msg}\n"
                "Run: ccproxy auth login claude_api"
            )

    if openai_oauth:
        authed, msg = check_ccproxy_auth("codex")
        if not authed:
            raise RuntimeError(
                f"ccproxy Codex OAuth not authenticated: {msg}\n"
                "Run: ccproxy auth login codex"
            )

    port = config.ccproxy_port
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid ccproxy port: {port}. Must be between 1 and 65535.")

    # Auto-patch ccproxy adapter to fix OAuth header compatibility
    _patch_ccproxy_oauth_header()

    # Start ccproxy (single process serves both providers)
    proc = ensure_ccproxy(
        port, anthropic_oauth=anthropic_oauth, openai_oauth=openai_oauth
    )

    # Set environment for each OAuth provider
    if anthropic_oauth:
        setup_ccproxy_env(port)
    if openai_oauth:
        setup_codex_env(port)

    if proc:
        logger.info("Started ccproxy on port %d", port)
    else:
        logger.info("Reusing existing ccproxy on port %d", port)
    return proc
