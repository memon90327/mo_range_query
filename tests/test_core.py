import math
import unittest

from mo_range_query import MoRangeQuery, AnswerAccumulator


class SumAccumulator(AnswerAccumulator):
    """Sum of arr[l..r]. add/remove are exact inverses, so ordering is free."""

    def __init__(self, arr):
        self._arr = arr
        self._s = 0

    def add_left(self, i):
        self._s += self._arr[i]

    def add_right(self, i):
        self._s += self._arr[i]

    def remove_left(self, i):
        self._s -= self._arr[i]

    def remove_right(self, i):
        self._s -= self._arr[i]

    def answer(self):
        return self._s


class DistinctAccumulator(AnswerAccumulator):
    """Count of distinct values in arr[l..r].

    Exercises the non-negative-counter invariant. Because Mo's algorithm
    expands before shrinking, the running count never goes negative mid-move;
    we assert that as an internal-consistency check.
    """

    def __init__(self, arr):
        self._arr = arr
        self._freq = {}
        self._distinct = 0

    def _add(self, i):
        v = self._arr[i]
        c = self._freq.get(v, 0)
        if c == 0:
            self._distinct += 1
        self._freq[v] = c + 1

    def _remove(self, i):
        v = self._arr[i]
        c = self._freq[v]
        assert c > 0, "counter underflowed; expand-before-shrink invariant violated"
        self._freq[v] = c - 1
        if c - 1 == 0:
            self._distinct -= 1

    def add_left(self, i):
        self._add(i)

    def add_right(self, i):
        self._add(i)

    def remove_left(self, i):
        self._remove(i)

    def remove_right(self, i):
        self._remove(i)

    def answer(self):
        return self._distinct


class TestConstruction(unittest.TestCase):
    def test_zero_n_permits_no_queries(self):
        mo = MoRangeQuery(n=0)
        self.assertEqual(mo.run(SumAccumulator([])), [])

    def test_negative_n_raises(self):
        with self.assertRaises(ValueError):
            MoRangeQuery(n=-1)

    def test_block_size_override_accepted(self):
        mo = MoRangeQuery(n=100, block_size=7)
        self.assertEqual(mo._compute_block_size(), 7)


class TestValidation(unittest.TestCase):
    def test_add_query_out_of_range_high(self):
        mo = MoRangeQuery(n=5)
        with self.assertRaises(ValueError):
            mo.add_query(0, 5)

    def test_add_query_out_of_range_low(self):
        mo = MoRangeQuery(n=5)
        with self.assertRaises(ValueError):
            mo.add_query(-1, 3)

    def test_add_query_l_greater_than_r(self):
        mo = MoRangeQuery(n=5)
        with self.assertRaises(ValueError):
            mo.add_query(3, 2)

    def test_single_element_query_at_end(self):
        mo = MoRangeQuery(n=5)
        mo.add_query(4, 4)
        self.assertEqual(mo.run(SumAccumulator([1, 2, 3, 4, 5])), [5])

    def test_full_range_query(self):
        mo = MoRangeQuery(n=5)
        mo.add_query(0, 4)
        self.assertEqual(mo.run(SumAccumulator([1, 2, 3, 4, 5])), [15])


class TestSum(unittest.TestCase):
    def test_several_queries(self):
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        mo = MoRangeQuery(n=len(arr))
        ids = [mo.add_query(l, r) for l, r in
               [(0, 10), (2, 7), (4, 4), (1, 3), (5, 9)]]
        self.assertEqual(ids, [0, 1, 2, 3, 4])
        expected = [sum(arr[l:r + 1]) for l, r in
                    [(0, 10), (2, 7), (4, 4), (1, 3), (5, 9)]]
        self.assertEqual(mo.run(SumAccumulator(arr)), expected)

    def test_queries_returned_in_insertion_order(self):
        arr = [10, 20, 30, 40, 50]
        mo = MoRangeQuery(n=len(arr))
        mo.add_query(1, 3)
        mo.add_query(0, 0)
        mo.add_query(2, 4)
        out = mo.run(SumAccumulator(arr))
        self.assertEqual(out, [90, 10, 120])

    def test_repeated_queries(self):
        arr = [1, 2, 3]
        mo = MoRangeQuery(n=len(arr))
        mo.add_query(0, 2)
        mo.add_query(0, 2)
        self.assertEqual(mo.run(SumAccumulator(arr)), [6, 6])

    def test_empty_query_set_returns_empty_list(self):
        mo = MoRangeQuery(n=10)
        self.assertEqual(mo.run(SumAccumulator([0] * 10)), [])


class TestDistinct(unittest.TestCase):
    def test_distinct_basic(self):
        arr = [1, 2, 1, 3, 2, 4, 1]
        mo = MoRangeQuery(n=len(arr))
        mo.add_query(0, 6)
        mo.add_query(0, 0)
        mo.add_query(1, 2)
        mo.add_query(2, 5)
        out = mo.run(DistinctAccumulator(arr))
        self.assertEqual(out, [4, 1, 2, 4])

    def test_distinct_all_same(self):
        arr = [7, 7, 7, 7, 7]
        mo = MoRangeQuery(n=len(arr))
        mo.add_query(0, 4)
        mo.add_query(1, 3)
        mo.add_query(2, 2)
        self.assertEqual(mo.run(DistinctAccumulator(arr)), [1, 1, 1])

    def test_distinct_non_negative_invariant_under_adversarial_order(self):
        # Queries chosen so the window shrinks a lot between consecutive
        # processed queries; the expand-before-shrink ordering must keep
        # the distinct counter non-negative at every step.
        arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mo = MoRangeQuery(n=len(arr))
        mo.add_query(0, 9)
        mo.add_query(0, 0)
        mo.add_query(5, 9)
        mo.add_query(2, 4)
        mo.add_query(0, 9)
        out = mo.run(DistinctAccumulator(arr))
        self.assertEqual(out, [10, 1, 5, 3, 10])


class TestLargerInput(unittest.TestCase):
    def test_matches_brute_force_on_pseudorandom_data(self):
        # Deterministic pseudorandom array/query set; no wall-clock dependence.
        n = 200
        arr = [(i * 37 + 11) % 23 for i in range(n)]
        mo = MoRangeQuery(n=n)
        queries = []
        rng_state = 12345
        for _ in range(60):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            a = rng_state % n
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            b = rng_state % n
            l, r = (a, b) if a <= b else (b, a)
            mo.add_query(l, r)
            queries.append((l, r))
        expected_distinct = []
        expected_sum = []
        for l, r in queries:
            expected_distinct.append(len(set(arr[l:r + 1])))
            expected_sum.append(sum(arr[l:r + 1]))
        self.assertEqual(mo.run(DistinctAccumulator(arr)), expected_distinct)
        self.assertEqual(mo.run(SumAccumulator(arr)), expected_sum)


class TestCustomBlockSize(unittest.TestCase):
    def test_block_size_one_still_correct(self):
        arr = [5, 3, 8, 6, 1, 9, 2]
        mo = MoRangeQuery(n=len(arr), block_size=1)
        mo.add_query(0, 6)
        mo.add_query(3, 5)
        mo.add_query(0, 0)
        self.assertEqual(mo.run(SumAccumulator(arr)),
                         [sum(arr), 16, 5])

    def test_block_size_larger_than_n(self):
        arr = [1, 2, 3]
        mo = MoRangeQuery(n=len(arr), block_size=100)
        mo.add_query(0, 2)
        mo.add_query(0, 1)
        self.assertEqual(mo.run(SumAccumulator(arr)), [6, 3])


if __name__ == "__main__":
    unittest.main()
