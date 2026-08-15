"""Hermetic defaults for Qt tests on developer machines and CI."""

from __future__ import annotations

import os


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
