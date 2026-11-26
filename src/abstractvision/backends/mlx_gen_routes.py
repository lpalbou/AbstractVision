"""MLX-Gen image routes: the engine's planner, and the capability-id table.

A request the planner routes (`uses_planner`) is checked by the backend against the engine's rows
for its model (`MFluxVisionBackend._image_route`), then given here to `resolve_generation_plan`, the
planner `mlxgen generate` uses, taken from the same engine import as the rows. The capability id it
selects is looked up in `IMAGE_ROUTES`, the one place that maps a route to what the backend runs.
Nothing here loads weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

from ..errors import CapabilityNotSupportedError, OptionalDependencyMissingError

# The strength of a latent image-to-image run whose request names none: the value AbstractVision
# 0.3.37 gave ERNIE's latent edits. mlx-gen's rows publish no strength default.
MLX_GEN_DEFAULT_IMAGE_STRENGTH = 0.4
# The row mode of a latent image-to-image route.
LATENT_IMAGE_TO_IMAGE_MODE = "latent-img2img"

# Families whose every image_to_image request is routed by the planner. Any other family is routed
# only when a request sets an option that `selects_route`.
PLANNED_IMAGE_TO_IMAGE_FAMILIES: FrozenSet[str] = frozenset({"flux2", "z-image", "qwen"})


@dataclass(frozen=True)
class RouteOption:
    """A request option the route selection reads.

    `flag`: the row field that says a route applies it (None: the rows' image counts decide).
    `selects_route`: setting it routes the request on any family. `families_0_3_37`: the families
    AbstractVision 0.3.37 applied it on, under some spelling; on a model the engine publishes no rows
    for, the option runs its 0.3.37 branch on those families and is refused on any other
    (`refused_without_rows`). `requires`: the option it only applies with; set without it, it is
    refused. `canvas`: a text option of an outpaint or a reframe (its box, fill or fill color).
    """

    name: str
    flag: Optional[str]
    selects_route: bool
    families_0_3_37: FrozenSet[str] = frozenset()
    requires: Optional[str] = None
    canvas: bool = False


# In the order a refusal names them.
ROUTE_OPTIONS: Tuple[RouteOption, ...] = (
    RouteOption(
        "image_strength",
        flag="supports_image_strength",
        selects_route=True,
        families_0_3_37=frozenset({"ernie-image"}),
    ),
    RouteOption("mask", flag="supports_mask", selects_route=False, families_0_3_37=frozenset({"qwen-edit", "fibo-edit"})),
    RouteOption(
        "reference_images", flag=None, selects_route=True, families_0_3_37=frozenset({"flux2", "qwen-edit"})
    ),
    RouteOption("outpaint_padding", flag="supports_outpaint", selects_route=True, canvas=True),
    RouteOption("reframe_padding", flag="supports_reframe", selects_route=True, canvas=True),
    RouteOption(
        "outpaint_fill", flag="supports_outpaint_fill", selects_route=True, requires="outpaint_padding", canvas=True
    ),
    RouteOption(
        "outpaint_fill_color",
        flag="supports_outpaint_fill",
        selects_route=True,
        requires="outpaint_padding",
        canvas=True,
    ),
    # A text_to_image option; 0.3.37 refused it on every model.
    RouteOption("control_image", flag="supports_control_image", selects_route=True),
)
# The text options of an outpaint or a reframe, and of them the boxes, which own the canvas.
CANVAS_OPTIONS: Tuple[str, ...] = tuple(option.name for option in ROUTE_OPTIONS if option.canvas)
CANVAS_BOXES: Tuple[str, ...] = tuple(
    option.name for option in ROUTE_OPTIONS if option.canvas and option.requires is None
)
# Options no route takes together (the engine's own pairs), refused with ValueError.
CONTRADICTING_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("mask", "image_strength"),
    ("mask", "outpaint_padding"),
    ("mask", "reframe_padding"),
    ("outpaint_padding", "reframe_padding"),
)
# The fill whose color `outpaint_fill_color` sets.
OUTPAINT_SOLID_FILL = "solid"


@dataclass(frozen=True)
class ImageRoute:
    """What the backend runs for one capability id.

    `variant` is the `_ModelKey.variant` of the instance it runs on, which decides the MLX-Gen class
    (`MFluxVisionBackend._ensure_model_impl`). `call` is "edit" (the 0.3.37 edit branch, which writes
    its own temporary files), "latent" (`_generate_run` with the source and a strength, or a mask
    instead of the strength) or "inpaint" (the class's own masked edit, the source and a mask); the routes outside the edit branch write their files into the run's
    directory. Whether the negative prompt is passed is the selected row's `supports_negative_prompt`
    on every route (`MFluxVisionBackend._takes_negative_prompt`).
    `keeps_request_guidance`: an unset guidance stays unset, so the engine class applies its default.
    "outpaint" and "reframe" run through MLX-Gen's canvas session, which owns the canvas keywords;
    "control" is text-to-image guided by a control image, with the row's ControlNet.
    """

    variant: str
    call: str
    keeps_request_guidance: bool = False


# capability_id -> route. The comment names the MLX-Gen class its variant builds.
IMAGE_ROUTES: Dict[str, ImageRoute] = {
    "flux2.edit": ImageRoute(variant="flux2-edit", call="edit"),  # Flux2KleinEdit
    "flux2.latent": ImageRoute(variant="default", call="latent"),  # Flux2Klein
    "flux2.inpaint": ImageRoute(  # Flux2KleinInpaint
        variant="flux2-inpaint", call="inpaint", keeps_request_guidance=True
    ),
    "flux2.multi-reference": ImageRoute(variant="flux2-edit", call="edit"),  # Flux2KleinEdit
    "flux2.outpaint": ImageRoute(  # Flux2KleinOutpaint
        variant="flux2-outpaint", call="outpaint", keeps_request_guidance=True
    ),
    "flux2.reframe": ImageRoute(  # Flux2KleinEdit
        variant="flux2-edit", call="reframe", keeps_request_guidance=True
    ),
    "ernie-image.latent": ImageRoute(variant="default", call="latent"),  # ErnieImageTurbo
    "qwen.latent": ImageRoute(variant="default", call="latent"),  # QwenImage
    "qwen.edit": ImageRoute(variant="default", call="edit"),  # QwenImageEdit
    "qwen.inpaint": ImageRoute(variant="default", call="edit"),  # QwenImageEdit
    "qwen.multi-reference": ImageRoute(variant="default", call="edit"),  # QwenImageEdit
    "z-image.latent": ImageRoute(variant="default", call="latent"),  # ZImage
    "z-image.inpaint": ImageRoute(variant="default", call="latent"),  # ZImage, with the mask
    # QwenImageControlNet; the instance's variant also names the row's ControlNet ("qwen-control:<model>").
    "qwen.control": ImageRoute(variant="qwen-control", call="control"),
    "qwen.outpaint": ImageRoute(variant="default", call="outpaint", keeps_request_guidance=True),  # QwenImageEdit
    "qwen.reframe": ImageRoute(variant="default", call="reframe", keeps_request_guidance=True),  # QwenImageEdit
}


@dataclass(frozen=True)
class SelectedImageRoute:
    """The route the planner selected for a request: its capability id, the table's entry, the row, the
    model selector the engine was asked with, and the plan's ControlNet (`repo:file`) when it names one."""

    capability_id: str
    route: ImageRoute
    row: Mapping[str, Any]
    selector: str
    control_model: Optional[str] = None


def uses_planner(*, task: str, family: Optional[str], options: FrozenSet[str]) -> bool:
    """Whether a request is routed by the engine's planner; otherwise it runs its 0.3.37 path."""
    if task == "text_to_image":
        return "control_image" in options
    if task != "image_to_image":
        return False
    return family in PLANNED_IMAGE_TO_IMAGE_FAMILIES or any(
        option.selects_route and option.name in options for option in ROUTE_OPTIONS
    )


def refused_without_rows(options: FrozenSet[str], *, family: Optional[str]) -> Optional[str]:
    """The first option a request sets that 0.3.37 did not apply on the model's family: what a model
    without rows refuses.

    None: the request runs its 0.3.37 path, as 0.3.37 ran it.
    """
    return next(
        (
            option.name
            for option in ROUTE_OPTIONS
            if option.name in options and family not in option.families_0_3_37
        ),
        None,
    )


def refuse_contradictions(options: FrozenSet[str]) -> None:
    """ValueError for two options no route takes together; CapabilityNotSupportedError for an option set
    without the one it applies with."""
    for first, second in CONTRADICTING_OPTIONS:
        if first in options and second in options:
            raise ValueError(f"{first} and {second} cannot be combined: no MLX-Gen route takes both.")
    for option in ROUTE_OPTIONS:
        if option.requires and option.name in options and option.requires not in options:
            raise CapabilityNotSupportedError(
                f"{option.name} applies only with {option.requires}.", field=option.name, supported=()
            )


def refuse_outpaint_fill(row: Mapping[str, Any], *, fill: Optional[str], fill_color: Optional[str]) -> None:
    """Refuse a fill the selected outpaint row does not list, and a fill color without the solid fill."""
    modes = tuple(str(mode) for mode in (row.get("outpaint_fill_modes") or ()))
    if fill is not None and fill not in modes:
        raise CapabilityNotSupportedError(
            f"MLX-Gen's {row.get('id')} route fills with {', '.join(modes)}, not {fill!r}.",
            field="outpaint_fill",
            supported=modes,
        )
    if fill_color is not None and fill != OUTPAINT_SOLID_FILL:
        raise CapabilityNotSupportedError(
            f"outpaint_fill_color applies only with outpaint_fill={OUTPAINT_SOLID_FILL!r}.",
            field="outpaint_fill_color",
            supported=(),
        )


def serves_flag(rows: Sequence[Mapping[str, Any]], flag: str) -> bool:
    """Whether one of the rows has `flag` and a capability id the backend runs."""
    return any(row.get(flag) and str(row.get("id")) in IMAGE_ROUTES for row in rows)


def serves_latent_route(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Whether one of a model's image_to_image rows is a latent route the backend runs."""
    return any(
        row.get("mode") == LATENT_IMAGE_TO_IMAGE_MODE and str(row.get("id")) in IMAGE_ROUTES for row in rows
    )


def resolve_image_route(
    resolve_generation_plan: Any,
    *,
    selector: str,
    rows: Sequence[Mapping[str, Any]],
    task: str,
    options: FrozenSet[str],
    reference_count: int,
    has_adapters: bool,
) -> SelectedImageRoute:
    """The route `resolve_generation_plan` (the engine's planner) selects for a request whose options
    passed the row checks.

    `has_lora` is never given to the planner, so an adapter does not change the row; adapters are
    refused afterwards when the selected row takes none. A combination no row serves, a route
    this release does not run, and adapters on such a row raise CapabilityNotSupportedError.
    """
    task_inference_error = _lazy_import_task_inference_error()
    has_image_strength = "image_strength" in options or _default_strength_applies(rows, options)
    try:
        plan = resolve_generation_plan(
            model=selector,
            task=task.replace("_", "-"),
            image_count=1 + reference_count if task == "image_to_image" else 0,
            has_image_strength=has_image_strength,
            has_mask="mask" in options,
            has_outpaint="outpaint_padding" in options,
            has_reframe="reframe_padding" in options,
            has_control_image="control_image" in options,
            has_lora=False,
        )
    except task_inference_error as e:
        raise CapabilityNotSupportedError(
            f"MLX-Gen has no {task} route for this combination of options: {e}", field=None, supported=None
        ) from None
    capability_id = str(plan.capability_id)
    row = next((row for row in rows if str(row.get("id")) == capability_id), {})
    route = IMAGE_ROUTES.get(capability_id)
    if route is None:
        field = _row_option(row)
        raise CapabilityNotSupportedError(
            f"This AbstractVision release does not run MLX-Gen's {capability_id} route"
            + (f" (it applies {field})." if field else "."),
            field=field,
            supported=() if field else None,
        )
    if has_adapters and not plan.supports_lora:
        raise CapabilityNotSupportedError(
            f"MLX-Gen's {capability_id} route takes no LoRA adapters for this model.",
            field="lora_adapters",
            supported=(),
        )
    control_model = str(plan.control_model) if getattr(plan, "control_model", None) else None
    return SelectedImageRoute(
        capability_id=capability_id, route=route, row=row, selector=selector, control_model=control_model
    )


def _default_strength_applies(rows: Sequence[Mapping[str, Any]], options: FrozenSet[str]) -> bool:
    # A plain edit (no option that selects another route) on a model whose one-image default row is
    # latent runs latent, at a strength.
    if options & {option.name for option in ROUTE_OPTIONS}:
        return False
    default = next((row for row in rows if row.get("default_for_task") and allows_image_count(row, 1)), None)
    return default is not None and default.get("mode") == LATENT_IMAGE_TO_IMAGE_MODE


def allows_image_count(row: Mapping[str, Any], count: int) -> bool:
    """Whether a row takes `count` input images (no maximum when it publishes none)."""
    maximum = row.get("max_images")
    return int(row.get("min_images") or 0) <= count and (maximum is None or count <= int(maximum))


def _row_option(row: Mapping[str, Any]) -> Optional[str]:
    # The request option a route exists for: its first set flag, else the references it takes.
    for option in ROUTE_OPTIONS:
        if option.flag and row.get(option.flag):
            return option.name
    return "reference_images" if allows_image_count(row, 2) else None


def _lazy_import_task_inference_error() -> Any:
    try:
        from mflux.task_inference import TaskInferenceError  # type: ignore
    except Exception as e:
        raise OptionalDependencyMissingError(
            "MLX-Gen route planning requires the MLX-Gen runtime. "
            'Install/upgrade it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    return TaskInferenceError
