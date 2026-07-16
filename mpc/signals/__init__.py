from typing import TypeVar

T = TypeVar("T")


def rotate_signal(signal: list[T], offset: int) -> list[T]:
    if offset >= 0:
        c = len(signal) - offset
    else:
        c = -offset

    return signal[c:] + signal[:c]


def extend_signal(signal: list[T], desired_length: int) -> list[T]:
    length = len(signal)

    if length > desired_length:
        raise ValueError(
            f"The signal is longer than the provided desired length: {desired_length} < {length}",
        )

    if length == desired_length:
        return signal

    last = signal[-1]
    return signal + [last] * (desired_length - length)
