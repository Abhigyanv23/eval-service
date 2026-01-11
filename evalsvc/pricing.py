"""Judge cost estimates. Prices are USD per million tokens (input, output).

Override or extend with EVALSVC_PRICING_JSON, e.g. '{"gpt-4.1": [2.0, 8.0]}'. Models that aren't listed
are reported as unpriced (cost None) rather than guessed."""
from __future__ import annotations

import json

from evalsvc.config import settings

# Anthropic first-party list prices.
_DEFAULT: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


def _table() -> dict[str, tuple[float, float]]:
    table = dict(_DEFAULT)
    if settings.pricing_json:
        table.update({k: (float(v[0]), float(v[1])) for k, v in json.loads(settings.pricing_json).items()})
    return table


PRICES = _table()


def cost_usd(judge_id: str, model: str | None, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated cost of the tokens. Local and heuristic judges are free; unknown paid models return None."""
    if judge_id.startswith(("heuristic", "local:")):
        return 0.0
    if model is None or model not in PRICES:
        return None
    p_in, p_out = PRICES[model]
    return round((input_tokens * p_in + output_tokens * p_out) / 1_000_000, 6)
