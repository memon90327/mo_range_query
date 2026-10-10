"""Mo's algorithm for offline batched range queries.

The processor reorders a set of queries by block decomposition so that the total
cost of moving the [l, r] window across the array is O((N + Q) * sqrt(N)) instead
of O(Q * N) for naive iteration. The user supplies a callback (an AnswerAccumulator)
that maintains incremental state as the window's left and right edges move by one
step at a time.

Design decisions (because the brief is intentionally short on detail):
  * Query ranges are inclusive [l, r]. This matches the canonical Mo's algorithm
    literature and keeps the expand/shrink logic symmetric.
  * The accumulator owns ALL state. This class never inspects or stores the array
    data; it only tracks the window and tells the accumulator when to add or remove
    an index. That keeps the core generic (sum, distinct-count, min-with-undo, etc.
    are all possible) without baking one semantics into the library.
  * We expand before shrinking inside move(). If the accumulator's add() is
    destructive or non-idempotent, the ordering expand-then-shrink avoids a
    transient state where an index is removed before its neighbour is added,
    which would underflow counters that can't go negative.
  * Ties in ordering are broken by the original query index so the result
    permutation is deterministic across Python versions and platforms.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


class AnswerAccumulator:
    """Protocol-style base for the incremental state object.

    Subclass this and implement the four methods. The core calls them in a
    specific order during a window move: expands (add) happen before shrinks
    (remove), which matters if your state cannot represent negative counts.
    """

    def add_left(self, index: int) -> None:
        """Incorporate the element at ``index`` into the window from the left side."""
        raise NotImplementedError

    def add_right(self, index: int) -> None:
        """Incorporate the element at ``index`` into the window from the right side."""
        raise NotImplementedError

    def remove_left(self, index: int) -> None:
        """Remove the element at ``index`` from the left edge of the window."""
        raise NotImplementedError

    def remove_right(self, index: int) -> None:
        """Remove the element at ``index`` from the right edge of the window."""
        raise NotImplementedError

    def answer(self) -> object:
        """Snapshot the current answer for the active window."""
        raise NotImplementedError


class MoRangeQuery:
    """Offline batched range-query processor using Mo's algorithm.

    Example::

        class SumAcc(AnswerAccumulator):
            def __init__(self, arr):
                self._arr = arr
                self._s = 0
            def add_left(self, i):  self._s += self._arr[i]
            def add_right(self, i): self._s += self._arr[i]
            def remove_left(self, i):  self._s -= self._arr[i]
            def remove_right(self, i): self._s -= self._arr[i]
            def answer(self): return self._s

        mo = MoRangeQuery(n=len(arr))
        mo.add_query(0, 4)
        mo.add_query(2, 7)
        results = mo.run(SumAcc(arr))  # [sum(arr[0..4]), sum(arr[2..7])]
    """

    def __init__(self, n: int, block_size: int = 0) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self._n = n
        self._queries: List[Tuple[int, int]] = []
        # block_size is computed lazily so that queries added later still benefit
        # from a size based on the final query count. Caller may override.
        self._block_size = block_size

    @property
    def n(self) -> int:
        return self._n

    def add_query(self, l: int, r: int) -> int:
        """Register an inclusive range query ``[l, r]``. Returns its 0-based id."""
        if not (0 <= l <= r < self._n):
            raise ValueError(
                f"query (l={l}, r={r}) out of bounds for n={self._n}; "
                "require 0 <= l <= r < n"
            )
        qid = len(self._queries)
        self._queries.append((l, r))
        return qid

    def _compute_block_size(self) -> int:
        """Determine block size.

        The canonical choice is sqrt(n); if there are very few queries relative
        to n, a larger block reduces overhead. We keep the simple sqrt choice
        because it is the proven bound and the caller can override.
        """
        if self._block_size > 0:
            return self._block_size
        if self._n <= 1:
            return 1
        return max(1, int(math.isqrt(self._n)))

    def _order_indices(self) -> List[int]:
        """Return query indices in Mo's processing order.

        Within the same block (queries whose l falls in [k*B, (k+1)*B)):
          - even blocks sort r ascending,
          - odd blocks sort r descending.
        This is the standard optimization that makes consecutive r moves
        sweep back and forth rather than restarting, halving the r-movement
        cost. Ties break by original index for deterministic output.
        """
        block = self._compute_block_size()
        order = list(range(len(self._queries)))
        order.sort(
            key=lambda i: (
                self._queries[i][0] // block,
                self._queries[i][1] if (self._queries[i][0] // block) % 2 == 0
                else -self._queries[i][1],
                i,
            )
        )
        return order

    def run(self, accumulator: AnswerAccumulator) -> List[object]:
        """Execute all registered queries in Mo's order.

        Returns answers aligned to the original ``add_query`` insertion order
        (result[i] answers the i-th query added).
        """
        if len(self._queries) == 0:
            return []

        order = self._order_indices()
        results: List[object] = [None] * len(self._queries)

        # Start with an empty window [0, -1] (length zero) so that the first
        # query expands from a known empty state rather than special-casing l=0.
        cur_l, cur_r = 0, -1

        for qi in order:
            target_l, target_r = self._queries[qi]
            self._move_window(accumulator, cur_l, cur_r, target_l, target_r)
            results[qi] = accumulator.answer()
            cur_l, cur_r = target_l, target_r

        return results

    @staticmethod
    def _move_window(
        acc: AnswerAccumulator, cur_l: int, cur_r: int, tgt_l: int, tgt_r: int
    ) -> None:
        """Move the window from [cur_l, cur_r] to [tgt_l, tgt_r].

        We always expand before shrinking. Concretely the four phases are:
          1. Expand right edge outwards (add).
          2. Expand left edge leftwards (add).
          3. Shrink left edge rightwards (remove).
          4. Shrink right edge leftwards (remove).
        If add() and remove() are inverses this is equivalent to any order, but
        if the accumulator tracks a non-negative counter (e.g. distinct element
        counts) then removing before adding could transiently drive the counter
        negative and break an invariant the accumulator relies on.
        """
        # Expand right: add indices (cur_r, tgt_r] when tgt_r > cur_r.
        while cur_r < tgt_r:
            cur_r += 1
            acc.add_right(cur_r)
        # Expand left: add indices [tgt_l, cur_l) when tgt_l < cur_l.
        while cur_l > tgt_l:
            cur_l -= 1
            acc.add_left(cur_l)
        # Shrink left: remove indices [cur_l, tgt_l) when tgt_l > cur_l.
        while cur_l < tgt_l:
            acc.remove_left(cur_l)
            cur_l += 1
        # Shrink right: remove indices (tgt_r, cur_r] when tgt_r < cur_r.
        while cur_r > tgt_r:
            acc.remove_right(cur_r)
            cur_r -= 1
