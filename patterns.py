import json
from itertools import combinations

import numpy as np
from scipy import ndimage

from geometry import evaluate_spec

SIZE = 512


def rectangle(mask, x0, y0, x1, y1):
    mask[y0:y1, x0:x1] = 1


def make_pattern(rng, kind, critical=None):

    image = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cx, cy = 256, 256
    if kind == "bridge":
        gap = int(rng.integers(8, 101)) if critical is None else int(critical)
        widths = rng.integers(100, 145, size=2).tolist()
        lengths = rng.integers(200, 337, size=2).tolist()
        offset = int(rng.integers(-30, 31))
        left_edge = cx - gap // 2
        right_edge = left_edge + gap
        rectangle(
            image,
            left_edge - widths[0],
            cy - lengths[0] // 2,
            left_edge,
            cy + lengths[0] // 2,
        )
        rectangle(
            image,
            right_edge,
            cy + offset - lengths[1] // 2,
            right_edge + widths[1],
            cy + offset + lengths[1] // 2,
        )
        points = [
            (cy, left_edge - widths[0] // 2),
            (cy + offset, right_edge + widths[1] // 2),
        ]
        params = dict(kind=kind, gap=gap, widths=widths, lengths=lengths, offset=offset)
    else:
        width = int(rng.integers(16, 113)) if critical is None else int(critical)
        neck_length = int(rng.integers(72, 177))
        pad_widths = rng.integers(104, 139, size=2).tolist()
        pad_heights = rng.integers(160, 221, size=2).tolist()
        offset = int(rng.integers(-24, 25))
        left_edge = cx - neck_length // 2
        right_edge = left_edge + neck_length
        rectangle(
            image,
            left_edge - pad_widths[0],
            cy - pad_heights[0] // 2,
            left_edge + 2,
            cy + pad_heights[0] // 2,
        )
        rectangle(
            image,
            right_edge - 2,
            cy + offset - pad_heights[1] // 2,
            right_edge + pad_widths[1],
            cy + offset + pad_heights[1] // 2,
        )
        rectangle(
            image, left_edge, cy - width // 2, right_edge, cy - width // 2 + width
        )
        points = [
            (cy, left_edge - pad_widths[0] // 2),
            (cy + offset, right_edge + pad_widths[1] // 2),
        ]
        params = dict(
            kind=kind,
            width=width,
            neck_length=neck_length,
            pad_widths=pad_widths,
            pad_heights=pad_heights,
            offset=offset,
        )

    clutter = []
    for _ in range(int(rng.integers(0, 4))):
        x = int(rng.choice([44, 414]))
        y = int(rng.integers(44, 433))
        w, h = int(rng.integers(24, 49)), int(rng.integers(24, 49))
        rectangle(image, x, y, x + w, y + h)
        clutter.append([x, y, w, h])
    params["clutter"] = clutter

    rotation = int(rng.integers(4))
    image = np.rot90(image, rotation).copy()
    for _ in range(rotation):
        points = [(SIZE - 1 - x, y) for y, x in points]
    dy, dx = rng.integers(-12, 13, size=2).tolist()
    image = np.roll(image, (dy, dx), axis=(0, 1))
    points = [(y + dy, x + dx) for y, x in points]
    spec = dict(
        kind=kind,
        connectivity=4,
        terminals=[[y - 4, x - 4, y + 5, x + 5] for y, x in points],
    )

    group_key = json.dumps(params, sort_keys=True, separators=(",", ":"))
    params.update(rotation=rotation, translation=[dy, dx])
    return image, spec, params, group_key


def select_specs(mask: np.ndarray, checks_per_kind: int = 2) -> list[dict]:

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Input layout must be two-dimensional")
    labels, _ = ndimage.label(mask, ndimage.generate_binary_structure(2, 1))
    depth = ndimage.distance_transform_edt(mask)
    components = []
    for component_id, bounding_box in enumerate(ndimage.find_objects(labels), 1):
        if bounding_box is None:
            continue
        local_mask = labels[bounding_box] == component_id
        coordinates = np.argwhere(local_mask)
        offset = np.array([bounding_box[0].start, bounding_box[1].start])
        coordinates += offset

        valid = (
            (coordinates[:, 0] >= 4)
            & (coordinates[:, 0] < mask.shape[0] - 4)
            & (coordinates[:, 1] >= 4)
            & (coordinates[:, 1] < mask.shape[1] - 4)
        )
        coordinates = coordinates[valid]
        if not len(coordinates):
            continue
        distances = depth[coordinates[:, 0], coordinates[:, 1]]
        if distances.max() < 5:
            continue
        centroid = coordinates.mean(axis=0)
        deepest = coordinates[distances == distances.max()]
        representative = deepest[np.argmin(((deepest - centroid) ** 2).sum(axis=1))]
        core = coordinates[distances >= max(5.0, 0.6 * distances.max())]
        first = core[np.argmax(((core - representative) ** 2).sum(axis=1))]
        second = core[np.argmax(((core - first) ** 2).sum(axis=1))]
        components.append(
            {
                "id": component_id,
                "representative": representative,
                "break_pair": [first, second],
                "span_squared": int(((first - second) ** 2).sum()),
            }
        )

    def make_spec(kind, points, component_ids, selection):
        terminals = [
            [int(y - 4), int(x - 4), int(y + 5), int(x + 5)] for y, x in points
        ]
        spec = {
            "kind": kind,
            "terminals": terminals,
            "connectivity": 4,
            "input_component_ids": component_ids,
            "selection": selection,
        }
        if evaluate_spec(mask, spec)["defect"]:
            raise RuntimeError("Input-only specification violates the input layout")
        return spec

    pairs = sorted(
        combinations(components, 2),
        key=lambda pair: int(
            ((pair[0]["representative"] - pair[1]["representative"]) ** 2).sum()
        ),
    )
    specs = []
    for first, second in pairs[:checks_per_kind]:
        specs.append(
            make_spec(
                "bridge",
                [first["representative"], second["representative"]],
                [first["id"], second["id"]],
                "nearest_distinct_component_representatives",
            )
        )
    for component in sorted(components, key=lambda item: -item["span_squared"])[
        :checks_per_kind
    ]:
        if component["span_squared"] < 100:
            continue
        specs.append(
            make_spec(
                "break",
                component["break_pair"],
                [component["id"], component["id"]],
                "farthest_interior_pair_of_large_span_component",
            )
        )
    return specs
