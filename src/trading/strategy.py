"""Trading-layer exports for backtest replay strategy adapters."""

from src.backtest.strategy import PureMarketMakerReplayAdapter as _PureMarketMakerReplayAdapter


class PureMarketMakerReplayAdapter(_PureMarketMakerReplayAdapter):
	"""Compatibility export for the dedicated pure-MM replay harness."""


__all__ = ["PureMarketMakerReplayAdapter"]
