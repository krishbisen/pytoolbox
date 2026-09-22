import pytest

from pytoolbox import DataStandardizer, chunked, clamp, is_palindrome


def test_clamp() -> None:
    assert clamp(15, 0, 10) == 10
    assert clamp(5, 0, 10) == 5


def test_clamp_rejects_invalid_range() -> None:
    with pytest.raises(ValueError):
        clamp(5, 10, 0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("level", True), ("Never odd or even", True), ("robot", False)],
)
def test_is_palindrome(text: str, expected: bool) -> None:
    assert is_palindrome(text) is expected


def test_chunked() -> None:
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunked_rejects_zero() -> None:
    with pytest.raises(ValueError):
        list(chunked([1, 2], 0))


def test_standardizer() -> None:
    standardizer = DataStandardizer()
    standardizer.fit([10, 20, 30])
    assert standardizer.transform([10, 20, 30]) == pytest.approx(
        [-1.22474487, 0.0, 1.22474487]
    )


def test_standardizer_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        DataStandardizer().transform([1, 2])
