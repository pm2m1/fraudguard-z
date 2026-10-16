"""Single-process WSGI entrypoint.

Start only behind an authenticated TLS-terminating reverse proxy bound to
a protected localhost or Unix socket listener. This module is not a server.
Use one worker: the SQLite + in-memory recipient history has one owner.
"""
from .serve import build_app

app = build_app(production=True)
