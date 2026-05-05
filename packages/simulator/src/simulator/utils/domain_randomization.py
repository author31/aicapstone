from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import AssetBase
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

DEFAULT_HDR_TEXTURES = [
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
    f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
]


def _randomize_domelight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    color_variation: float,
    textures: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    new_intensity = random.uniform(intensity_range[0], intensity_range[1])
    light_prim.GetAttribute("inputs:intensity").Set(new_intensity)

    offsets = [random.uniform(-color_variation, color_variation) for _ in range(3)]
    avg = sum(offsets) / 3
    new_color = tuple(max(0.0, min(1.0, 0.75 + o - avg)) for o in offsets)
    light_prim.GetAttribute("inputs:color").Set(new_color)

    if textures:
        light_prim.GetAttribute("inputs:texture:file").Set(random.choice(textures))


def randomize_light_conditions(
    name: str = "light",
    intensity_range: tuple[float, float] = (1500.0, 3000.0),
    color_variation: float = 0.4,
    textures: list[str] | None = None,
) -> EventTerm:
    return EventTerm(
        func=_randomize_domelight,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "intensity_range": intensity_range,
            "color_variation": color_variation,
            "textures": textures if textures is not None else DEFAULT_HDR_TEXTURES,
        },
    )
