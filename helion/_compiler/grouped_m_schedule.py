from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING
from typing import Sequence

import torch

from .. import exc

if TYPE_CHECKING:
    from .compile_environment import CompileEnvironment
    from .device_ir import GraphInfo


@dataclasses.dataclass(frozen=True)
class GroupedMSchedulePlan:
    jagged_block_id: int
    parent_block_ids: tuple[int, ...]
    group_block_id: int
    offsets: torch.Tensor
    loop_graph_ids: tuple[int, ...]

    def owns_block_id(self, block_id: int) -> bool:
        return block_id == self.jagged_block_id or block_id in self.parent_block_ids


def collect_grouped_m_schedule_plans(
    env: CompileEnvironment,
    graphs: Sequence[GraphInfo],
) -> tuple[GroupedMSchedulePlan, ...]:
    plans: list[GroupedMSchedulePlan] = []
    for jagged_block_id, info in env.jagged_tile_schedule_infos.items():
        if info.schedule is None:
            continue
        if info.schedule != "grouped_m":
            raise exc.InvalidJaggedTileUsage(
                f"unsupported jagged tile schedule {info.schedule!r}"
            )
        if info.group_id is None:
            raise exc.InvalidJaggedTileUsage(
                "grouped_m jagged tile schedule is missing group block id"
            )
        if info.group_id not in info.parent_ids:
            raise exc.InvalidJaggedTileUsage(
                "grouped_m jagged tile group block id must be a parent block id"
            )
        if info.offsets is None:
            raise exc.InvalidJaggedTileUsage(
                "grouped_m jagged tile schedule is missing offsets"
            )
        if info.offsets.ndim != 1 or info.offsets.dtype != torch.int32:
            raise exc.InvalidJaggedTileUsage(
                "grouped_m jagged tile offsets must be a rank-1 int32 tensor"
            )

        related_blocks = set(info.parent_ids)
        related_blocks.add(jagged_block_id)
        loop_graph_ids = tuple(
            graph.graph_id
            for graph in graphs
            if related_blocks.intersection(getattr(graph, "block_ids", ()))
        )
        plans.append(
            GroupedMSchedulePlan(
                jagged_block_id=jagged_block_id,
                parent_block_ids=info.parent_ids,
                group_block_id=info.group_id,
                offsets=info.offsets,
                loop_graph_ids=loop_graph_ids,
            )
        )
    return tuple(plans)
