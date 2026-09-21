"""A minimal generic plugin registry.

Every extensibility point in this package (hardware providers, model
providers, runtime adapters, compatibility rules) is "just" a list of
objects satisfying some protocol, held in one of these. Adding support for
a new GPU vendor, model family, or engine means registering a new plugin
instance, never editing a branch of central if/else logic.
"""

from __future__ import annotations

from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """An ordered collection of plugins, queried by predicate rather than name.

    Order matters: providers are tried in registration order and the first
    one that claims it can handle the input wins. Built-in providers should
    generally be registered before third-party ones so the latter can
    override behavior by registering earlier via :meth:`register` with
    ``priority=True``.
    """

    def __init__(self) -> None:
        self._items: list[T] = []

    def register(self, item: T, *, priority: bool = False) -> None:
        """Add a plugin. ``priority=True`` inserts it at the front."""
        if priority:
            self._items.insert(0, item)
        else:
            self._items.append(item)

    def unregister(self, item: T) -> None:
        self._items.remove(item)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def all(self) -> list[T]:
        return list(self._items)
