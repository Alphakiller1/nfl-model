"""NFL projections with an explicit authority gate.

Two forecasts share one authority. `forecast_slate` is the market-anchored
moneyline contract `profit-priority` consumes; `project_game` is the points
model -- margin, total and scoreline -- that the board and the dashboard render.
Both are gated by `authority`, and neither may emit a bet while the published
evidence says the model does not beat the closing line.
"""

from .authority import Action, Authority, Level, current, promote
from .forecast import (
    DEFAULT_LAMBDA,
    SPREAD_LAMBDA,
    GameForecast,
    GameProjection,
    anchor,
    forecast_game,
    forecast_slate,
    project_game,
)
from .market import PairedQuote, american_to_implied, devig_two_way, prob_to_american
from .matrix import TeamForm

__version__ = "0.2.0"
__all__ = [
    "DEFAULT_LAMBDA",
    "SPREAD_LAMBDA",
    "Action",
    "Authority",
    "GameForecast",
    "GameProjection",
    "Level",
    "PairedQuote",
    "TeamForm",
    "american_to_implied",
    "anchor",
    "current",
    "devig_two_way",
    "forecast_game",
    "forecast_slate",
    "prob_to_american",
    "project_game",
    "promote",
]
