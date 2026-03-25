from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SyncGateTelemetry:
    contagion_sync_blocks: int = 0
    si9_sync_blocks: int = 0
    si10_sync_blocks: int = 0

    def record_contagion_block(self) -> None:
        self.contagion_sync_blocks += 1

    def record_si9_block(self) -> None:
        self.si9_sync_blocks += 1

    def record_si10_block(self) -> None:
        self.si10_sync_blocks += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "contagion_sync_blocks": int(self.contagion_sync_blocks),
            "si9_sync_blocks": int(self.si9_sync_blocks),
            "si10_sync_blocks": int(self.si10_sync_blocks),
        }