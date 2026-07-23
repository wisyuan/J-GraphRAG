"""J-Lens graph-build extraction modules (concepts / roles / entities /
relations / filters).

All modules are import-safe: importing them touches no disk, model, or GPU.
torch / transformers / jlens / sklearn are imported lazily inside functions.
"""
from . import concepts, entities, filter, relations, roles

__all__ = ["concepts", "entities", "filter", "relations", "roles"]
