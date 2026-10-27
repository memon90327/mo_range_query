# mo_range_query

Offline processor for batched range queries that reorders queries by block decomposition to achieve O((N+Q)√N) total time, instead of O(Q·N) for naive per-query iteration.

## Usage

```python
from mo_range_query import MoRangeQuery, AnswerAccumulator

class DistinctCount(AnswerAccumulator):
    def __init__(self, arr):
        self._arr = arr
        self._freq = {}
        self._distinct = 0
    def _add(self, i):
        v = self._arr[i]
        if self._freq.get(v, 0) == 0:
            self._distinct += 1
        self._freq[v] = self._freq.get(v, 0) + 1
    def _remove(self, i):
        v = self._arr[i]
        self._freq[v] -= 1
        if self._freq[v] == 0:
            self._distinct -= 1
    def add_left(self, i):  self._add(i)
    def add_right(self, i): self._add(i)
    def remove_left(self, i):  self._remove(i)
    def remove_right(self, i): self._remove(i)
    def answer(self): return self._distinct

arr = [1, 2, 1, 3, 2, 4, 1]
mo = MoRangeQuery(n=len(arr))
mo.add_query(0, 6)  # returns query id 0
mo.add_query(2, 5)  # returns query id 1
print(mo.run(DistinctCount(arr)))  # [4, 4], aligned to insertion order
```

Exported names: `MoRangeQuery`, `AnswerAccumulator`.

## Why this exists

You have a fixed array and a known-in-advance batch of range queries, and each query's answer can be maintained incrementally as the range's left and right edges move by one position. Mo's algorithm reorders the queries so the total movement of the window is O((N+Q)√N) rather than O(Q·N). The classic use case is "count of distinct elements in each range", but anything that supports add-one-index / remove-one-index works (sum, frequency of a fixed value, min-with-undo, etc.).

The trade-off: answers are produced in the original insertion order, but internally they are computed in Mo's order, so you cannot stream queries in one at a time — you must register the full batch before calling `run()`. This is an offline algorithm by nature.

## Edge cases worth knowing

- Query ranges are **inclusive** `[l, r]` with `0 <= l <= r < n`. `add_query(0, 0)` is a single-element query and is legal.
- `n = 0` is permitted; `run()` returns `[]`.
- The accumulator is driven by the core in a specific order: **expansions happen before shrinks** within each window move. If your accumulator tracks a non-negative counter (distinct-count, frequency), this ordering keeps the counter from transiently going negative mid-move. You do not need to handle negative counts defensively, but your `remove_*` methods must not be called on an index that is not currently in the window — the core guarantees this never happens.
- The block size defaults to `√n`; pass `block_size=` to override (useful if you want a fixed schedule for benchmarking, not for correctness).

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

