from dataclasses import dataclass
import pytest

from mpc.signals import extend_signal, rotate_signal


def test_rotate_signal() -> None:
    @dataclass
    class TestCase:
        signal: list[int]
        offset: int
        expected: list[int]

        def run(self) -> None:
            actual = rotate_signal(self.signal, self.offset)
            assert (
                actual == self.expected
            ), f"signal: {self.signal}, offset: {self.offset}"

    for test in [
        TestCase([1, 2, 3], 0, [1, 2, 3]),
        TestCase([1, 2, 3], 1, [3, 1, 2]),
        TestCase([1, 2, 3, 4, 5], 3, [3, 4, 5, 1, 2]),
        TestCase([1, 2, 3], -1, [2, 3, 1]),
        TestCase([1, 2, 3, 4, 5], -2, [3, 4, 5, 1, 2]),
    ]:
        test.run()


def test_extend_signal() -> None:
    signal = [0.0, 1.0]

    with pytest.raises(ValueError):
        extend_signal(signal, 1)
    assert extend_signal(signal, 2) == [0.0, 1.0]
    assert extend_signal(signal, 3) == [0.0, 1.0, 1.0]
