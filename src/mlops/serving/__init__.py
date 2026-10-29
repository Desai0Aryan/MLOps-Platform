def create_app(*args, **kwargs):
    """Lazy import: keeps `mlops.serving.db` importable without pulling in FastAPI/experiments."""
    from .app import create_app as _create_app
    return _create_app(*args, **kwargs)
