"""Undo stack — keeps a bounded history of inverse operations."""

from contextlib import contextmanager
from typing import Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Undo stack — keeps a bounded history of inverse operations.
# ---------------------------------------------------------------------------

class UndoStack:
    """A small stack of (description, undo_callable, redo_callable) tuples.

    Each mutation in CocoState pushes an entry here. Ctrl+Z pops and runs
    the undo callable; Ctrl+Shift+Z / Ctrl+Y pops and runs the redo callable
    (the redo stack is cleared on any new mutation).

    Use ``with stack.group("...")`` around a multi-step mutation (e.g.
    discard-all) so it undoes/redoes as a single entry.

    Callables are zero-arg. They run on the main thread synchronously.
    """

    MAX_DEPTH = 100

    def __init__(self) -> None:
        self._undo: List[Tuple[str, Any, Any]] = []
        self._redo: List[Tuple[str, Any, Any]] = []
        # When not None, push() collects entries here instead of pushing
        # them directly; group() flushes the batch as one composite entry.
        self._batch: Optional[List[Tuple[str, Any, Any]]] = None
        # While > 0, push() drops entries (see mute()).
        self._muted: int = 0

    def push(self, description: str, undo: Any, redo: Any) -> None:
        """Push an (undo, redo) pair. Clears the redo stack."""
        if self._muted:
            self._redo.clear()
            return
        if self._batch is not None:
            self._batch.append((description, undo, redo))
            self._redo.clear()
            return
        self._undo.append((description, undo, redo))
        if len(self._undo) > self.MAX_DEPTH:
            self._undo.pop(0)
        self._redo.clear()

    @contextmanager
    def mute(self):
        """Drop pushes inside the block — for mutations whose undo entries
        the caller will re-push itself as one composite (e.g. a background
        propagation run whose per-frame pushes must not swallow unrelated
        user edits into an open batch). Redo is still cleared: a mutation
        happened."""
        self._muted += 1
        try:
            yield
        finally:
            self._muted -= 1

    @contextmanager
    def group(self, description: str):
        """Coalesce every push inside the block into one undo/redo entry.

        Nested groups merge into the outermost one. Undo runs the collected
        undo callables in reverse order; redo runs them in original order.
        """
        outer = self._batch
        self._batch = []
        try:
            yield
        finally:
            entries = self._batch
            self._batch = outer
        if not entries:
            return
        if outer is not None:
            outer.extend(entries)
            return

        def undo_all(entries=tuple(entries)) -> None:
            for _desc, undo, _redo in reversed(entries):
                undo()

        def redo_all(entries=tuple(entries)) -> None:
            for _desc, _undo, redo in entries:
                redo()

        self.push(description, undo_all, redo_all)

    def pop_undo(self) -> Optional[Tuple[str, Any, Any]]:
        if not self._undo:
            return None
        entry = self._undo.pop()
        # Move to redo stack so Ctrl+Shift+Z can re-apply.
        self._redo.append(entry)
        return entry

    def pop_redo(self) -> Optional[Tuple[str, Any, Any]]:
        if not self._redo:
            return None
        entry = self._redo.pop()
        self._undo.append(entry)
        return entry

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
