from __future__ import annotations

from config import (
    BOX_COUNT,
    EXPORT_IMAGE,
    EXPORT_IMAGE_PATH,
    EXPORT_HTML,
    EXPORT_HTML_PATH,
    EXPORT_JSON,
    EXPORT_JSON_PATH,
    PALLET_LENGTH_MM,
    PALLET_MAX_HEIGHT_MM,
    PALLET_WIDTH_MM,
    USE_GA,
    VIS_DROP_FRAMES,
    VIS_FRAME_MS,
    VIS_FLOAT_Z_OFFSET_MM,
)
from engine import pack_boxes
from ga import pack_from_individual, run_evolution
from generator import generate_boxes
from models import Pallet
from visualization import (
    export_manifest,
    export_plotly_animation,
    export_three_view_image,
)


def main() -> None:
    boxes = generate_boxes(BOX_COUNT)
    pallet = Pallet(
        length_mm=PALLET_LENGTH_MM,
        width_mm=PALLET_WIDTH_MM,
        height_mm=PALLET_MAX_HEIGHT_MM,
    )

    if USE_GA:
        best, logbook = run_evolution(
            boxes,
            (PALLET_LENGTH_MM, PALLET_WIDTH_MM, PALLET_MAX_HEIGHT_MM),
        )
        pallet, placed, skipped = pack_from_individual(
            best,
            boxes,
            (PALLET_LENGTH_MM, PALLET_WIDTH_MM, PALLET_MAX_HEIGHT_MM),
        )
        print(f"Best fitness: {best.fitness.values[0]:.4f}")
        if logbook:
            last = logbook[-1]
            print(
                f"GA stats (gen {last['gen']}): min={last['min']:.4f}, "
                f"avg={last['avg']:.4f}, max={last['max']:.4f}"
            )
    else:
        placed, skipped = pack_boxes(boxes, pallet)

    pallet_dims = (pallet.length_mm, pallet.width_mm, pallet.height_mm)
    if EXPORT_JSON:
        export_manifest(placed, EXPORT_JSON_PATH)
        print(f"Wrote manifest: {EXPORT_JSON_PATH}")
    if EXPORT_HTML:
        export_plotly_animation(
            placed,
            pallet_dims,
            EXPORT_HTML_PATH,
            frame_ms=VIS_FRAME_MS,
            frames_per_box=VIS_DROP_FRAMES,
            float_offset_mm=VIS_FLOAT_Z_OFFSET_MM,
        )
        print(f"Wrote visualization: {EXPORT_HTML_PATH}")
    if EXPORT_IMAGE:
        export_three_view_image(placed, pallet_dims, EXPORT_IMAGE_PATH)
        print(f"Wrote image: {EXPORT_IMAGE_PATH}")

    print(f"Placed {len(placed)} boxes, skipped {len(skipped)} boxes")
    if placed:
        max_height = max(b.position[2] + b.rotated_dims()[2] for b in placed)
        print(f"Max stack height: {max_height} mm")


if __name__ == "__main__":
    main()
