"""Market-anchored NFL forecasts with an explicit authority gate."""

from .authority import Action, Authority, Level, current, promote
from .forecast import DEFAULT_LAMBDA, GameForecast, anchor, forecast_game, forecast_slate
from .market import PairedQuote, american_to_implied, devig_two_way, prob_to_american
from .projections import division_winners, week_one_projections

__version__ = "0.1.0"
__all__ = [
    "DEFAULT_LAMBDA",
    "Action",
    "Authority",
    "GameForecast",
    "Level",
    "PairedQuote",
    "american_to_implied",
    "anchor",
    "current",
    "devig_two_way",
    "division_winners",
    "forecast_game",
    "forecast_slate",
    "prob_to_american",
    "promote",
    "week_one_projections",
]
