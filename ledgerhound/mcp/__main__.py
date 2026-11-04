"""python -m ledgerhound.mcp -- serve Ledgerhound over MCP stdio."""

import sys

from ledgerhound.mcp._stdio import protect_stdout

protect_stdout()  # before anything else can print

from ledgerhound.mcp.server import main  # noqa: E402

sys.exit(main())
