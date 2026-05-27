# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.
"""Enables `python -m modelmeld.cli setup ...`. Delegates to cli.main()."""
import sys

from modelmeld.cli import main

sys.exit(main())
