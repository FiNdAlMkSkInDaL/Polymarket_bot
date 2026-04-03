from __future__ import annotations

from importlib import import_module

_EXPORT_MAP = {
	"OFIMomentumDetector": ("src.signals.ofi_momentum", "OFIMomentumDetector"),
	"OFIMomentumSignal": ("src.signals.ofi_momentum", "OFIMomentumSignal"),
	"ExhaustionFader": ("src.signals.exhaustion_fader", "ExhaustionFader"),
	"HybridArbMaker": ("src.signals.hybrid_arb_maker", "HybridArbMaker"),
	"LongTailMarketMaker": ("src.signals.long_tail_maker", "LongTailMarketMaker"),
	"ObiScalper": ("src.signals.obi_scalper", "ObiScalper"),
	"VacuumMaker": ("src.signals.vacuum_maker", "VacuumMaker"),
	"WallJumper": ("src.signals.wall_jumper", "WallJumper"),
	"ContagionArbDetector": ("src.signals.contagion_arb", "ContagionArbDetector"),
	"ContagionArbSignal": ("src.signals.contagion_arb", "ContagionArbSignal"),
	"CtfPegDetector": ("src.signals.ctf_peg_detector", "CtfPegDetector"),
	"CtfPegState": ("src.signals.ctf_peg_detector", "CtfPegState"),
	"DisputeArbitrageDetector": ("src.signals.dispute_arbitrage_detector", "DisputeArbitrageDetector"),
	"ShadowSweepDetector": ("src.signals.shadow_sweep_detector", "ShadowSweepDetector"),
	"ShadowSweepSignal": ("src.signals.shadow_sweep_detector", "ShadowSweepSignal"),
}

__all__ = list(_EXPORT_MAP)


def __getattr__(name: str):
	export = _EXPORT_MAP.get(name)
	if export is None:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	module_name, attribute_name = export
	value = getattr(import_module(module_name), attribute_name)
	globals()[name] = value
	return value


def __dir__() -> list[str]:
	return sorted(list(globals().keys()) + __all__)
