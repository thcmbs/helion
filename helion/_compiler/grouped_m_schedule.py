from __future__ import annotations

import dataclasses
import enum
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


class GroupedMDimRole(enum.Enum):
    PACKED_M = "packed_m"
    GROUP = "group"
    TILE = "tile"
    FULL_SLICE = "full_slice"
    SCALAR = "scalar"
    UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class GroupedMTensorAccessRole:
    graph_id: int
    node_name: str
    kind: str
    tensor: torch.Tensor
    dim_roles: tuple[GroupedMDimRole, ...]


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


def collect_grouped_m_access_roles(
    env: CompileEnvironment,
    graphs: Sequence[GraphInfo],
    plans: Sequence[GroupedMSchedulePlan],
) -> tuple[GroupedMTensorAccessRole, ...]:
    if not plans:
        return ()

    from ..language import memory_ops

    role_by_tensor_id: dict[int, GroupedMTensorAccessRole] = {}
    for graph in graphs:
        for node in graph.graph.nodes:
            if node.op != "call_function" or node.target not in (
                memory_ops.load,
                memory_ops.store,
            ):
                continue
            tensor_arg = node.args[0]
            subscript = node.args[1]
            if not isinstance(tensor_arg, torch.fx.Node) or not isinstance(
                subscript, (list, tuple)
            ):
                continue
            tensor = tensor_arg.meta.get("val")
            if not isinstance(tensor, torch.Tensor):
                continue

            for plan in plans:
                dim_roles = _classify_subscript_roles(env, plan, subscript)
                if GroupedMDimRole.PACKED_M not in dim_roles:
                    continue
                role = GroupedMTensorAccessRole(
                    graph_id=graph.graph_id,
                    node_name=node.name,
                    kind="store" if node.target is memory_ops.store else "load",
                    tensor=tensor,
                    dim_roles=dim_roles,
                )
                role_by_tensor_id[id(tensor)] = role
                break
    return tuple(role_by_tensor_id.values())


def _classify_subscript_roles(
    env: CompileEnvironment,
    plan: GroupedMSchedulePlan,
    subscript: Sequence[object],
) -> tuple[GroupedMDimRole, ...]:
    roles: list[GroupedMDimRole] = []
    for idx in subscript:
        if idx is None:
            continue
        if isinstance(idx, slice):
            roles.append(
                GroupedMDimRole.FULL_SLICE
                if idx == slice(None)
                else GroupedMDimRole.UNKNOWN
            )
            continue

        value = idx.meta.get("val") if isinstance(idx, torch.fx.Node) else idx
        block_ids = _block_ids_in_value(env, value)
        if plan.group_block_id in block_ids and plan.jagged_block_id in block_ids:
            roles.append(GroupedMDimRole.PACKED_M)
        elif plan.group_block_id in block_ids:
            roles.append(GroupedMDimRole.GROUP)
        elif block_ids:
            roles.append(GroupedMDimRole.TILE)
        elif isinstance(value, (int, torch.SymInt)):
            roles.append(GroupedMDimRole.SCALAR)
        else:
            # TODO(grouped-m): support original flattened 1D accesses such as
            # packed_m * stride + tile_n via an explicit FlatPackedAffinePattern.
            roles.append(GroupedMDimRole.UNKNOWN)
    return tuple(roles)


def _block_ids_in_value(env: CompileEnvironment, value: object) -> frozenset[int]:
    block_ids: set[int] = set()
    if isinstance(value, torch.Tensor):
        for dim_size in value.shape:
            block_id = env.resolve_block_id(dim_size)
            if isinstance(block_id, int):
                block_ids.add(block_id)
    elif isinstance(value, torch.SymInt):
        block_id = env.resolve_block_id(value)
        if isinstance(block_id, int):
            block_ids.add(block_id)
    return frozenset(block_ids)
