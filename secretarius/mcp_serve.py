from __future__ import annotations

from .mcp_server import run_stdio_server


def main() -> int:
    run_stdio_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
