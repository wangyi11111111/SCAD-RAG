"""Setuptools shim for editable installs in restricted Windows workspaces."""

from __future__ import annotations

import tempfile

from setuptools import setup


setup(options={"egg_info": {"egg_base": tempfile.gettempdir()}})
