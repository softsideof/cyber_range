# Contributing

Pull requests welcome. If you're adding a new scenario, include tests.

## Setup

```bash
pip install -e ".[dev,inference]"
python -m pytest tests/ -v
ruff check . && ruff format .
```

## Adding a new scenario

1. Add the scenario config in `server/attack_engine.py` (see `SCENARIOS` dict)
2. Update `openenv.yaml` with the task definition
3. Add tests in `tests/test_scenarios.py`
4. Update baseline scores in the README
