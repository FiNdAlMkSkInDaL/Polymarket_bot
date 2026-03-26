from __future__ import annotations

import pytest

from src.signals.mm_tracker import MarketMakerTracker
from src.simulation.mm_algo_simulator import SimulatedMarketMaker


def test_tracker_converges_on_hidden_market_maker_kappa_in_closed_loop() -> None:
    target = SimulatedMarketMaker(
        "wintermute_cluster",
        true_kappa=0.00005,
        initial_mid_price=0.5,
        noise_std=0.00001,
        random_seed=11,
    )
    radar = MarketMakerTracker(alpha=0.2)

    previous_mid = target.mid_price
    for _ in range(50):
        new_mid = target.receive_taker_flow(100.0)
        price_delta = new_mid - previous_mid
        radar.process_fill_event(target.maker_address, price_delta, 100.0)
        previous_mid = new_mid

    fingerprint = radar.fingerprints[target.maker_address]
    assert fingerprint.kappa_ewma == pytest.approx(target.true_kappa, rel=0.05)

    target_spread_delta = 0.02
    predicted_attack_volume = fingerprint.calculate_attack_volume(target_spread_delta)
    exact_attack_volume = target_spread_delta / target.true_kappa

    assert predicted_attack_volume == pytest.approx(exact_attack_volume, rel=0.05)

    vulnerable = radar.get_vulnerable_makers(
        target_spread_delta=target_spread_delta,
        max_capital=predicted_attack_volume * 1.01,
    )
    assert vulnerable == [
        (target.maker_address, pytest.approx(predicted_attack_volume)),
    ]

    noiseless_target = SimulatedMarketMaker(
        "wintermute_cluster_noiseless",
        true_kappa=target.true_kappa,
        initial_mid_price=0.5,
        noise_std=0.0,
        random_seed=11,
    )
    before_attack_mid = noiseless_target.mid_price
    after_attack_mid = noiseless_target.receive_taker_flow(predicted_attack_volume)
    realized_spread_delta = after_attack_mid - before_attack_mid

    assert realized_spread_delta == pytest.approx(target_spread_delta, rel=0.05)