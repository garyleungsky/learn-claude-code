---
name: testing
description: Write and run tests for Python projects. Use when user asks to test code, write unit tests, or debug failing tests.
tags: python, pytest, testing
---

# Testing Skill

You now have expertise in Python testing. Follow these workflows:

## Writing Tests with pytest

**Basic test structure:**
```python
import pytest

def test_function_returns_expected():
    result = my_function(1, 2)
    assert result == 3

def test_function_raises_on_invalid_input():
    with pytest.raises(ValueError):
        my_function("bad", "input")
```

**Test class for grouping related tests:**
```python
class TestMyClass:
    def setup_method(self):
        self.obj = MyClass()

    def test_default_state(self):
        assert self.obj.value == 0

    def test_update(self):
        self.obj.update(42)
        assert self.obj.value == 42
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_example.py

# Run specific test
pytest tests/test_example.py::test_function_name

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Show print output
pytest -s

# Run tests matching a keyword
pytest -k "test_login"
```

## Fixtures

```python
import pytest

@pytest.fixture
def sample_data():
    return {"name": "test", "value": 42}

@pytest.fixture
def tmp_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    return f

def test_with_fixtures(sample_data, tmp_file):
    assert sample_data["name"] == "test"
    assert tmp_file.read_text() == "hello world"
```

## Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert input * 2 == expected
```

## Mocking

```python
from unittest.mock import patch, MagicMock

def test_api_call():
    with patch("module.requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"status": "ok"}
        result = my_api_function()
        assert result["status"] == "ok"
        mock_get.assert_called_once()
```

## Best Practices

1. **Name tests descriptively** - `test_login_fails_with_wrong_password`
2. **One assertion per test** when possible
3. **Use fixtures** for shared setup, not copy-paste
4. **Test edge cases** - empty inputs, None, boundary values
5. **Keep tests fast** - mock external dependencies
6. **Run tests before committing** - `pytest -x` to catch failures early
