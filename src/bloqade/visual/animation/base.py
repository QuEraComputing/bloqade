import itertools
import dataclasses
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Tuple, Callable, Optional

import matplotlib.patches as mpatches
import matplotlib.collections as mcollections

from .gate_event import GateEvent

_gate_color_code = {
    "GlobalCZGate": "tab:red",
    "TopHatCZGate": "tab:red",
    "GlobalHyperfineRotation": "tab:blue",
    "GlobalHyperfineZRotation": "tab:green",
    "LocalHyperfineZRotation": "tab:green",
    "LocalHyperfineZRotationPos": "tab:green",
    "LocalHyperfineRotation": "tab:blue",
    "LocalHyperfineRotationPos": "tab:blue",
}


class quera_color_code(str, Enum):
    # TODO: for python 3.11+, replace traits with StrEnum
    purple = "#6437FF"
    red = "#C2477F"
    yellow = "#EADD08"


@dataclasses.dataclass
class GateArtist:
    mpl_ax: Any

    @abstractmethod
    def clear_data(self) -> None:
        pass

    @abstractmethod
    def get_artists(self) -> Tuple[Any]:
        pass


@dataclasses.dataclass(init=False)
class GlobalGateArtist(GateArtist):
    mpl_obj: mpatches.Rectangle

    def __init__(self, mpl_ax: Any, xmin, ymin, width, height, color):
        super().__init__(mpl_ax)
        rc = mpatches.Rectangle(
            [xmin, ymin], width, height, color=color, alpha=0.6, visible=False
        )
        mpl_ax.add_patch(rc)
        self.mpl_obj = rc

    def clear_data(self) -> None:
        self.mpl_obj.set_width(0)
        self.mpl_obj.set_height(0)
        self.mpl_obj.set_xy([0, 0])

    def get_artists(self) -> Tuple[Any]:
        return (self.mpl_obj,)

    def set_visible(self, visible: bool):
        self.mpl_obj.set_visible(visible)


@dataclasses.dataclass(init=False)
class RowRegionGateArtist(GateArtist):
    """
    A row region gate artist object.

    bound box is [y_origin - width/2, y_origin + width/2]
    """

    mpl_obj: mpatches.Rectangle
    mpl_obj_keepout_top: mpatches.Rectangle
    mpl_obj_keepout_btm: mpatches.Rectangle
    width: float
    xmin: float

    def __init__(
        self, mpl_ax: Any, xmin, width, ymin, ymin_keepout, ymax, ymax_keepout, color
    ):
        super().__init__(mpl_ax)
        self.width = width
        self.xmin = xmin
        rc_btm = mpatches.Rectangle(
            [xmin, ymin_keepout],
            width,
            ymin - ymin_keepout,
            color=color,
            alpha=0.3,
            visible=False,
        )
        mpl_ax.add_patch(rc_btm)
        self.mpl_obj_keepout_btm = rc_btm

        rc = mpatches.Rectangle(
            [xmin, ymin], width, ymax - ymin, color=color, alpha=0.6, visible=False
        )
        mpl_ax.add_patch(rc)
        self.mpl_obj = rc

        rc_top = mpatches.Rectangle(
            [xmin, ymax],
            width,
            ymax_keepout - ymax,
            color=color,
            alpha=0.3,
            visible=False,
        )
        mpl_ax.add_patch(rc_top)
        self.mpl_obj_keepout_top = rc_top

    def clear_data(self) -> None:
        self.mpl_obj.set_width(0)
        self.mpl_obj.set_height(0)
        self.mpl_obj.set_xy([0, 0])

        self.mpl_obj_keepout_top.set_width(0)
        self.mpl_obj_keepout_top.set_height(0)
        self.mpl_obj_keepout_top.set_xy([0, 0])

        self.mpl_obj_keepout_btm.set_width(0)
        self.mpl_obj_keepout_btm.set_height(0)
        self.mpl_obj_keepout_btm.set_xy([0, 0])

    def get_artists(self) -> Tuple[Any]:
        return (self.mpl_obj, self.mpl_obj_keepout_top, self.mpl_obj_keepout_btm)

    def update_data(self, ymin, ymax, ymin_keepout, ymax_keepout):
        self.mpl_obj.set_height(ymax - ymin)
        self.mpl_obj.set_width(self.width)
        self.mpl_obj.set_xy([self.xmin, ymin])

        self.mpl_obj_keepout_top.set_height(ymax_keepout - ymax)
        self.mpl_obj_keepout_top.set_width(self.width)
        self.mpl_obj_keepout_top.set_xy([self.xmin, ymax])

        self.mpl_obj_keepout_btm.set_height(ymin - ymin_keepout)
        self.mpl_obj_keepout_btm.set_width(self.width)
        self.mpl_obj_keepout_btm.set_xy([self.xmin, ymin_keepout])

    def set_visible(self, visible: bool):
        self.mpl_obj.set_visible(visible)
        self.mpl_obj_keepout_btm.set_visible(visible)
        self.mpl_obj_keepout_top.set_visible(visible)


@dataclasses.dataclass
class SpotGateArtist(GateArtist):
    mpl_obj: mcollections.PathCollection

    def __init__(self, mpl_ax: Any, xy: List[Tuple[float, float]], radius, color):
        super().__init__(mpl_ax)
        self.mpl_obj = mpl_ax.scatter(
            [x for x, y in xy],
            [y for x, y in xy],
            s=radius,
            visible=False,
            marker="o",
            facecolors=color,
            alpha=0.5,
            zorder=+100,
        )

    def clear_data(self) -> None:
        self.mpl_obj.set_offsets([])

    def get_artists(self) -> Tuple[Any]:
        return (self.mpl_obj,)

    def update_data(self, xy: List[Tuple[float, float]]):
        self.mpl_obj.set_offsets(xy)

    def set_visible(self, visible: bool):
        self.mpl_obj.set_visible(visible)


@dataclasses.dataclass
class FieldOfView:
    xmin: Optional[float] = dataclasses.field(default=None)
    xmax: Optional[float] = dataclasses.field(default=None)
    ymin: Optional[float] = dataclasses.field(default=None)
    ymax: Optional[float] = dataclasses.field(default=None)

    def not_defined(self):
        return (
            self.xmin is None
            or self.xmax is None
            or self.ymin is None
            or self.ymax is None
        )

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    def to_json(self):
        return {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
        }

    @classmethod
    def from_json(cls, json_dict):
        return FieldOfView(
            xmin=json_dict["xmin"],
            xmax=json_dict["xmax"],
            ymin=json_dict["ymin"],
            ymax=json_dict["ymax"],
        )


@dataclasses.dataclass(init=False)
class GatePainter:
    artist_objs: Dict[str, GateArtist]
    artist_methods: Dict[str, Callable]

    def __init__(self, mpl_ax, qpu_fov: FieldOfView, scale: float = 1.0):
        # gates:
        self.artist_objs = {}
        self.artist_methods = {
            "GlobalCZGate": self.process_global_cz_gate,
            "GlobalHyperfineRotation": self.process_global_hyperfine_rotation,
            "TopHatCZGate": self.process_tophat_cz_gate,
            "LocalHyperfineZRotation": self.process_local_hyperfine_z_rotation,
            "LocalHyperfineZRotationPos": self.process_local_hyperfine_z_rotation_pos,
            "LocalHyperfineRotation": self.process_local_hyperfine_rotation,
            "LocalHyperfineRotationPos": self.process_local_hyperfine_rotation_pos,
        }
        self.artist_objs["GlobalCZGate"] = GlobalGateArtist(
            mpl_ax=mpl_ax,
            xmin=qpu_fov.xmin,
            width=qpu_fov.width,
            ymin=qpu_fov.ymin,
            height=qpu_fov.height,
            color=_gate_color_code["GlobalCZGate"],
        )
        self.artist_objs["GlobalHyperfineRotation"] = GlobalGateArtist(
            mpl_ax=mpl_ax,
            xmin=qpu_fov.xmin,
            width=qpu_fov.width,
            ymin=qpu_fov.ymin,
            height=qpu_fov.height,
            color=_gate_color_code["GlobalHyperfineRotation"],
        )
        self.artist_objs["TopHatCZGate"] = RowRegionGateArtist(
            mpl_ax=mpl_ax,
            xmin=qpu_fov.xmin,
            width=qpu_fov.width,
            ymin=0,
            ymin_keepout=0,
            ymax=0,
            ymax_keepout=0,
            color=_gate_color_code["TopHatCZGate"],
        )

        self.artist_objs["LocalHyperfineZRotation"] = SpotGateArtist(
            mpl_ax=mpl_ax,
            xy=[],
            radius=160 * scale,
            color=_gate_color_code["LocalHyperfineZRotation"],
        )
        self.artist_objs["LocalHyperfineZRotationPos"] = SpotGateArtist(
            mpl_ax=mpl_ax,
            xy=[],
            radius=160 * scale,
            color=_gate_color_code["LocalHyperfineZRotationPos"],
        )

        self.artist_objs["LocalHyperfineRotation"] = SpotGateArtist(
            mpl_ax=mpl_ax,
            xy=[],
            radius=80,
            color=_gate_color_code["LocalHyperfineRotation"],
        )
        self.artist_objs["LocalHyperfineRotationPos"] = SpotGateArtist(
            mpl_ax=mpl_ax,
            xy=[],
            radius=80,
            color=_gate_color_code["LocalHyperfineRotationPos"],
        )

    @staticmethod
    def process_global_cz_gate(artist_obj, **kwargs) -> None:
        artist_obj.set_visible(True)

    @staticmethod
    def process_global_hyperfine_rotation(artist_obj, **kwargs) -> None:
        artist_obj.set_visible(True)

    @staticmethod
    def process_tophat_cz_gate(artist_obj, **kwargs) -> None:
        artist_obj.update_data(
            ymin=kwargs["y_min"],
            ymax=kwargs["y_max"],
            ymin_keepout=kwargs["y_min_keepout"],
            ymax_keepout=kwargs["y_max_keepout"],
        )
        artist_obj.set_visible(True)

    @staticmethod
    def process_local_hyperfine_z_rotation_pos(artist_obj, **kwargs) -> None:
        artist_obj.update_data(xy=[(x, kwargs["y_row_loc"]) for x in kwargs["varargs"]])
        artist_obj.set_visible(True)

    @staticmethod
    def process_local_hyperfine_rotation_pos(artist_obj, **kwargs) -> None:
        artist_obj.update_data(xy=[(x, kwargs["y_row_loc"]) for x in kwargs["varargs"]])
        artist_obj.set_visible(True)

    @staticmethod
    def process_local_hyperfine_z_rotation(artist_obj, **kwargs) -> None:
        raise ValueError(
            "cannot have local HyperfineZRotation, it should be already lowered to Pos variant!"
        )

    @staticmethod
    def process_local_hyperfine_rotation(artist_obj, **kwargs) -> None:
        raise ValueError(
            "cannot have local HyperfineRotation, it should be already lowered to Pos variant!"
        )

    def process_gates(self, gates: List[GateEvent]) -> List:

        for gate_artist in self.artist_objs.values():
            gate_artist.set_visible(False)

        for gate in gates:
            self.artist_methods[gate.cls_name](
                self.artist_objs[gate.cls_name], **gate.kwargs
            )

        return list(
            itertools.chain.from_iterable(
                gate_artist.get_artists() for gate_artist in self.artist_objs.values()
            )
        )
