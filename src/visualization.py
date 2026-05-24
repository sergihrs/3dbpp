from __future__ import annotations

import json
import math
from typing import List, Sequence, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models import Box

COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]

AXIS_Z_MAX_MM = 1500
EDGE_EPS_MM = 0.0
MESH_INSET_MM = 1.0
ARM_LINE_COLOR = "#2B2B2B"
ARM_RING_COLOR = "#1F1F1F"
ARM_LINE_WIDTH = 6
ARM_RING_WIDTH = 4

BOX_FACES = [
    (0, 1, 2),
    (0, 2, 3),
    (4, 5, 6),
    (4, 6, 7),
    (0, 1, 5),
    (0, 5, 4),
    (1, 2, 6),
    (1, 6, 5),
    (2, 3, 7),
    (2, 7, 6),
    (3, 0, 4),
    (3, 4, 7),
]
BOX_I, BOX_J, BOX_K = zip(*BOX_FACES)


def build_manifest(placed: Sequence[Box]) -> List[dict]:
    steps: List[dict] = []
    for step, box in enumerate(placed, start=1):
        if box.position is None:
            continue
        x, y, z = box.position
        dx, dy, dz = box.rotated_dims()
        steps.append(
            {
                "step": step,
                "box_id": box.box_id,
                "position_mm": {"x": int(x), "y": int(y), "z": int(z)},
                "dims_mm": {"x": int(dx), "y": int(dy), "z": int(dz)},
            }
        )
    return steps


def export_manifest(placed: Sequence[Box], output_path: str) -> List[dict]:
    manifest = build_manifest(placed)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def _floor_trace(length_mm: int, width_mm: int) -> go.Mesh3d:
    vertices = [
        (0, 0, 0),
        (length_mm, 0, 0),
        (length_mm, width_mm, 0),
        (0, width_mm, 0),
    ]
    xs, ys, zs = zip(*vertices)
    faces = [(0, 1, 2), (0, 2, 3)]
    i, j, k = zip(*faces)

    return go.Mesh3d(
        x=list(xs),
        y=list(ys),
        z=list(zs),
        i=list(i),
        j=list(j),
        k=list(k),
        color="#E6E6E6",
        opacity=0.18,
        flatshading=True,
        name="Pallet",
        hoverinfo="skip",
    )


def _mesh_coords(
    x: float, y: float, z: float, dx: float, dy: float, dz: float
) -> Tuple[List[float], List[float], List[float]]:
    vertices = [
        (x, y, z),
        (x + dx, y, z),
        (x + dx, y + dy, z),
        (x, y + dy, z),
        (x, y, z + dz),
        (x + dx, y, z + dz),
        (x + dx, y + dy, z + dz),
        (x, y + dy, z + dz),
    ]
    xs, ys, zs = zip(*vertices)
    return list(xs), list(ys), list(zs)


def _mesh_coords_inset(
    x: float, y: float, z: float, dx: float, dy: float, dz: float, inset_mm: float
) -> Tuple[List[float], List[float], List[float]]:
    inset = max(0.0, inset_mm)
    if dx <= 2 * inset or dy <= 2 * inset or dz <= 2 * inset:
        return _mesh_coords(x, y, z, dx, dy, dz)
    return _mesh_coords(
        x + inset,
        y + inset,
        z + inset,
        dx - 2 * inset,
        dy - 2 * inset,
        dz - 2 * inset,
    )


def _edge_coords(
    x: float, y: float, z: float, dx: float, dy: float, dz: float
) -> Tuple[List[float], List[float], List[float]]:
    eps = max(0.0, EDGE_EPS_MM)
    x0 = x - eps
    y0 = y - eps
    z0 = z - eps
    x1 = x + dx + eps
    y1 = y + dy + eps
    z1 = z + dz + eps
    p0 = (x0, y0, z0)
    p1 = (x1, y0, z0)
    p2 = (x1, y1, z0)
    p3 = (x0, y1, z0)
    p4 = (x0, y0, z1)
    p5 = (x1, y0, z1)
    p6 = (x1, y1, z1)
    p7 = (x0, y1, z1)

    edges = [
        (p0, p1),
        (p1, p2),
        (p2, p3),
        (p3, p0),
        (p4, p5),
        (p5, p6),
        (p6, p7),
        (p7, p4),
        (p0, p4),
        (p1, p5),
        (p2, p6),
        (p3, p7),
    ]
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for a, b in edges:
        xs.extend([a[0], b[0], None])
        ys.extend([a[1], b[1], None])
        zs.extend([a[2], b[2], None])
    return xs, ys, zs


def _axis_cm(max_mm: int, title: str) -> dict:
    step_mm = 100
    tickvals = list(range(0, max_mm + 1, step_mm))
    ticktext = [str(int(v / 10)) for v in tickvals]
    return dict(
        range=[0, max_mm],
        autorange=False,
        title=title,
        tickvals=tickvals,
        ticktext=ticktext,
        ticks="outside",
        showbackground=False,
        gridcolor="#D0D0D0",
        zerolinecolor="#B5B5B5",
        showspikes=False,
    )


def _circle_points(cx: float, cy: float, z: float, radius: float, steps: int = 32) -> Tuple[List[float], List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for i in range(steps + 1):
        theta = (2.0 * math.pi * i) / steps
        xs.append(cx + radius * math.cos(theta))
        ys.append(cy + radius * math.sin(theta))
        zs.append(z)
    return xs, ys, zs


def _aspect_ratio(x_len: float, y_len: float, z_len: float) -> dict:
    max_len = max(x_len, y_len, z_len, 1.0)
    return {"x": x_len / max_len, "y": y_len / max_len, "z": z_len / max_len}


def _pose_from_box(box: Box) -> Tuple[float, float, float, float, float, float, int]:
    if box.position is None:
        raise ValueError("Box has no position")
    x, y, z = box.position
    dx, dy, dz = box.rotated_dims()
    return float(x), float(y), float(z), float(dx), float(dy), float(dz), box.box_id


def _box_trace_from_pose(
    pose: Tuple[float, float, float, float, float, float, int],
    color: str,
    visible: bool,
) -> go.Mesh3d:
    x, y, z, dx, dy, dz, box_id = pose
    xs, ys, zs = _mesh_coords_inset(x, y, z, dx, dy, dz, MESH_INSET_MM)
    custom = [[box_id, int(x), int(y), int(z), int(dx), int(dy), int(dz)]] * 8

    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=list(BOX_I),
        j=list(BOX_J),
        k=list(BOX_K),
        color=color,
        opacity=1.0,
        flatshading=True,
        lighting=dict(ambient=0.7, diffuse=0.6, specular=0.2, roughness=0.9),
        lightposition=dict(x=1000, y=1000, z=2000),
        name=f"Box {box_id}",
        customdata=custom,
        hovertemplate=(
            "Box %{customdata[0]}<br>"
            "Pos: (%{customdata[1]}, %{customdata[2]}, %{customdata[3]}) mm<br>"
            "Dims: (%{customdata[4]}, %{customdata[5]}, %{customdata[6]}) mm"
            "<extra></extra>"
        ),
        visible=visible,
    )


def _box_edges_trace_from_pose(
    pose: Tuple[float, float, float, float, float, float, int],
    visible: bool,
) -> go.Scatter3d:
    x, y, z, dx, dy, dz, _ = pose
    xs, ys, zs = _edge_coords(x, y, z, dx, dy, dz)
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color="#111111", width=5),
        hoverinfo="skip",
        visible=visible,
    )


def _arm_traces_from_pose(
    pose: Tuple[float, float, float, float, float, float, int],
    z_top: float,
    visible: bool,
    z_axis_max: float,
) -> Tuple[go.Scatter3d, go.Scatter3d]:
    x, y, z, dx, dy, dz, _ = pose
    cx = x + dx / 2.0
    cy = y + dy / 2.0
    top_z = z_top
    ring_z = min(top_z, z_axis_max - 1.0)
    arm_z0 = z_axis_max
    if arm_z0 <= ring_z:
        arm_z0 = ring_z + 1.0
    arm_line = go.Scatter3d(
        x=[cx, cx],
        y=[cy, cy],
        z=[arm_z0, ring_z],
        mode="lines",
        line=dict(color=ARM_LINE_COLOR, width=ARM_LINE_WIDTH),
        hoverinfo="skip",
        visible=visible,
    )

    radius = max(20.0, min(120.0, min(dx, dy) * 0.25))
    ring_xs, ring_ys, ring_zs = _circle_points(cx, cy, ring_z, radius)
    arm_ring = go.Scatter3d(
        x=ring_xs,
        y=ring_ys,
        z=ring_zs,
        mode="lines",
        line=dict(color=ARM_RING_COLOR, width=ARM_RING_WIDTH),
        hoverinfo="skip",
        visible=visible,
    )
    return arm_line, arm_ring


def _max_stack_height(poses: Sequence[Tuple[float, float, float, float, float, float, int]]) -> float:
    if not poses:
        return 0.0
    return max(z + dz for _, _, z, _, _, dz, _ in poses)


def build_animation_figure(
    placed: Sequence[Box],
    pallet_dims: Tuple[int, int, int],
    frame_ms: int,
    frames_per_box: int,
    float_offset_mm: float,
) -> go.Figure:
    length_mm, width_mm, height_mm = pallet_dims
    poses = [_pose_from_box(box) for box in placed]
    z_axis_max = AXIS_Z_MAX_MM

    floor = _floor_trace(length_mm, width_mm)
    box_traces = [
        _box_trace_from_pose(pose, COLORS[idx % len(COLORS)], visible=False)
        for idx, pose in enumerate(poses)
    ]
    edge_traces = [_box_edges_trace_from_pose(pose, visible=False) for pose in poses]
    arm_line_traces = [
        _arm_traces_from_pose(pose, pose[2] + pose[5], visible=False, z_axis_max=z_axis_max)[0]
        for pose in poses
    ]
    arm_ring_traces = [
        _arm_traces_from_pose(pose, pose[2] + pose[5], visible=False, z_axis_max=z_axis_max)[1]
        for pose in poses
    ]
    fig = go.Figure(data=[floor] + box_traces + edge_traces + arm_line_traces + arm_ring_traces)

    frames_per_box = max(1, frames_per_box)
    if frames_per_box == 1:
        t_values = [1.0]
    else:
        t_values = [i / (frames_per_box - 1) for i in range(frames_per_box)]

    if poses:
        x0, y0, z0, dx0, dy0, dz0, _ = poses[0]
        start_z = max(height_mm, z0 + dz0) + float_offset_mm
        start_z = min(start_z, max(0.0, z_axis_max - dz0))
        init_t = t_values[1] if len(t_values) > 1 else t_values[0]
        cur_z_init = start_z + (z0 - start_z) * init_t
        xs, ys, zs = _mesh_coords_inset(x0, y0, cur_z_init, dx0, dy0, dz0, MESH_INSET_MM)
        box_traces[0].update(x=xs, y=ys, z=zs, visible=True)
        xs, ys, zs = _edge_coords(x0, y0, cur_z_init, dx0, dy0, dz0)
        edge_traces[0].update(x=xs, y=ys, z=zs, visible=True)
        arm_line, arm_ring = _arm_traces_from_pose(
            poses[0],
            cur_z_init + dz0,
            True,
            z_axis_max,
        )
        arm_line_traces[0].update(x=arm_line.x, y=arm_line.y, z=arm_line.z, visible=True)
        arm_ring_traces[0].update(x=arm_ring.x, y=arm_ring.y, z=arm_ring.z, visible=True)

    aspect_ratio = _aspect_ratio(length_mm, width_mm, z_axis_max)
    fig.update_layout(
        title=dict(
            text="Palletization sequence",
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        margin=dict(l=24, r=24, t=64, b=28),
        showlegend=False,
        paper_bgcolor="#FAFAFA",
        font=dict(family="Space Grotesk, Avenir Next, sans-serif", size=12, color="#222222"),
        scene=dict(
            xaxis=_axis_cm(length_mm, "X (cm)"),
            yaxis=_axis_cm(width_mm, "Y (cm)"),
            zaxis=_axis_cm(z_axis_max, "Z (cm)"),
            aspectmode="manual",
            aspectratio=aspect_ratio,
            bgcolor="#FAFAFA",
        ),
        uirevision="pallet-animation",
        dragmode="turntable",
    )
    fig.update_scenes(camera=dict(eye=dict(x=1.6, y=1.6, z=1.0), up=dict(x=0, y=0, z=1)))

    if not poses:
        return fig

    trace_indices = list(
        range(1, 1 + len(box_traces) + len(edge_traces) + len(arm_line_traces) + len(arm_ring_traces))
    )
    frames: List[go.Frame] = []

    for idx, pose in enumerate(poses):
        x, y, z, dx, dy, dz, _ = pose
        start_z = max(height_mm, z + dz) + float_offset_mm
        start_z = min(start_z, max(0.0, z_axis_max - dz))

        t_values_box = t_values
        if idx == 0 and len(t_values) > 1:
            t_values_box = t_values[1:]

        for sub_idx, t in enumerate(t_values_box, start=1):
            cur_z = start_z + (z - start_z) * t
            mesh_frame_data: List[go.Mesh3d] = []
            edge_frame_data: List[go.Scatter3d] = []
            arm_line_data: List[go.Scatter3d] = []
            arm_ring_data: List[go.Scatter3d] = []
            for jdx, pose_j in enumerate(poses):
                pjx, pjy, pjz, pjdx, pjdy, pjdz, _ = pose_j
                if jdx < idx:
                    visible = True
                    pose_out = (pjx, pjy, pjz, pjdx, pjdy, pjdz, pose_j[6])
                elif jdx == idx:
                    visible = True
                    pose_out = (pjx, pjy, cur_z, pjdx, pjdy, pjdz, pose_j[6])
                else:
                    visible = False
                    pose_out = (pjx, pjy, pjz, pjdx, pjdy, pjdz, pose_j[6])

                mesh_frame_data.append(
                    _box_trace_from_pose(
                        pose_out,
                        COLORS[jdx % len(COLORS)],
                        visible,
                    )
                )
                edge_frame_data.append(_box_edges_trace_from_pose(pose_out, visible))

                if jdx == idx and visible:
                    top_z = pose_out[2] + pose_out[5]
                    arm_line, arm_ring = _arm_traces_from_pose(pose_out, top_z, True, z_axis_max)
                else:
                    arm_line, arm_ring = _arm_traces_from_pose(pose_out, pose_out[2] + pose_out[5], False, z_axis_max)
                arm_line_data.append(arm_line)
                arm_ring_data.append(arm_ring)

            frames.append(
                go.Frame(
                    data=mesh_frame_data + edge_frame_data + arm_line_data + arm_ring_data,
                    traces=trace_indices,
                    name=f"{idx + 1}-{sub_idx}",
                )
            )

    fig.frames = frames

    slider_steps = []
    for idx in range(len(poses)):
        step_sub = frames_per_box
        if idx == 0 and frames_per_box > 1:
            step_sub = frames_per_box - 1
        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [f"{idx + 1}-{step_sub}"],
                    {
                        "mode": "immediate",
                        "frame": {"duration": frame_ms, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
                label=str(idx + 1),
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                bgcolor="#FFFFFF",
                bordercolor="#C0C0C0",
                borderwidth=1,
                font=dict(size=12, color="#222222"),
                pad=dict(l=8, r=8, t=4, b=4),
                x=0.0,
                y=1.02,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": frame_ms, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=slider_steps,
                x=0.08,
                y=0.0,
                len=0.92,
                bgcolor="#F0F0F0",
                bordercolor="#C0C0C0",
                borderwidth=1,
                activebgcolor="#DADADA",
                font=dict(size=11),
            )
        ],
    )

    return fig


def build_three_view_figure(
    placed: Sequence[Box],
    pallet_dims: Tuple[int, int, int],
) -> go.Figure:
    length_mm, width_mm, height_mm = pallet_dims
    poses = [_pose_from_box(box) for box in placed]
    z_axis_max = AXIS_Z_MAX_MM
    aspect_ratio = _aspect_ratio(length_mm, width_mm, z_axis_max)

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Front", "Top", "Side"),
    )

    for col in range(1, 4):
        fig.add_trace(_floor_trace(length_mm, width_mm), row=1, col=col)
        for idx, pose in enumerate(poses):
            fig.add_trace(
                _box_trace_from_pose(pose, COLORS[idx % len(COLORS)], visible=True),
                row=1,
                col=col,
            )
            fig.add_trace(
                _box_edges_trace_from_pose(pose, visible=True),
                row=1,
                col=col,
            )

    fig.update_layout(
        margin=dict(l=24, r=24, t=60, b=20),
        showlegend=False,
        paper_bgcolor="#FAFAFA",
        font=dict(family="Space Grotesk, Avenir Next, sans-serif", size=12, color="#222222"),
        scene=dict(
            xaxis=_axis_cm(length_mm, "X (cm)"),
            yaxis=_axis_cm(width_mm, "Y (cm)"),
            zaxis=_axis_cm(z_axis_max, "Z (cm)"),
            aspectmode="manual",
            aspectratio=aspect_ratio,
            camera=dict(eye=dict(x=0.0, y=-2.3, z=0.6), up=dict(x=0, y=0, z=1)),
            bgcolor="#FAFAFA",
        ),
        scene2=dict(
            xaxis=_axis_cm(length_mm, "X (cm)"),
            yaxis=_axis_cm(width_mm, "Y (cm)"),
            zaxis=_axis_cm(z_axis_max, "Z (cm)"),
            aspectmode="manual",
            aspectratio=aspect_ratio,
            camera=dict(eye=dict(x=0.0, y=0.0, z=2.6), up=dict(x=0, y=0, z=1)),
            bgcolor="#FAFAFA",
        ),
        scene3=dict(
            xaxis=_axis_cm(length_mm, "X (cm)"),
            yaxis=_axis_cm(width_mm, "Y (cm)"),
            zaxis=_axis_cm(z_axis_max, "Z (cm)"),
            aspectmode="manual",
            aspectratio=aspect_ratio,
            camera=dict(eye=dict(x=-2.3, y=0.0, z=0.6), up=dict(x=0, y=0, z=1)),
            bgcolor="#FAFAFA",
        ),
    )

    return fig


def export_plotly_animation(
    placed: Sequence[Box],
    pallet_dims: Tuple[int, int, int],
    output_path: str,
    frame_ms: int = 200,
    frames_per_box: int = 6,
    float_offset_mm: float = 300.0,
) -> str:
    fig = build_animation_figure(
        placed,
        pallet_dims,
        frame_ms,
        frames_per_box,
        float_offset_mm,
    )
    fig.write_html(output_path, include_plotlyjs=True, full_html=True, auto_open=False)
    return output_path


def export_three_view_image(
    placed: Sequence[Box],
    pallet_dims: Tuple[int, int, int],
    output_path: str,
) -> str:
    fig = build_three_view_figure(placed, pallet_dims)
    fig.write_image(output_path, width=1800, height=600, scale=2)
    return output_path
