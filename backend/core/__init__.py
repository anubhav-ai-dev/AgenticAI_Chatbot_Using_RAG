"""Application core: settings, database, shared exceptions and the DI container.

Keep this package initialiser empty — importing `backend.core.container` here would
create an import cycle (services import `core.config` / `core.exceptions`).
"""
