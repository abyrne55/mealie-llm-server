#!/usr/bin/python3
"""Container healthcheck — stdlib-only, no shell required."""

import sys
import urllib.request

try:
    urllib.request.urlopen("http://localhost:8000/health", timeout=5)
except Exception:
    sys.exit(1)
