from __future__ import annotations

import multiprocessing as mp
import os
import random
from functools import partial
from typing import List, Sequence, Tuple

from deap import algorithms, base, creator, tools

from config import (
    GA_CXPB,
    GA_MUTPB,
    GA_NGEN,
    GA_ORDER_MUTPB,
    GA_POOL_PROCESSES,
    GA_POP_SIZE,
    GA_ROT_MUTPB,
    GA_SKIP_PENALTY,
    GA_TOURN_SIZE,
    GA_VERBOSE,
    RANDOM_SEED,
)
from engine import pack_boxes_with_rotations
from models import Box, Pallet


def _ensure_creator() -> None:
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def _make_individual(box_count: int) -> List[List[int]]:
    order = random.sample(range(box_count), box_count)
    rotations = [random.randint(0, 5) for _ in range(box_count)]
    return [order, rotations]


def _mate(ind1: List[List[int]], ind2: List[List[int]]) -> Tuple[List, List]:
    tools.cxPartialyMatched(ind1[0], ind2[0])
    tools.cxTwoPoint(ind1[1], ind2[1])
    return ind1, ind2


def _mutate(
    individual: List[List[int]], order_indpb: float, rot_indpb: float
) -> Tuple[List[List[int]]]:
    tools.mutShuffleIndexes(individual[0], indpb=order_indpb)
    for i in range(len(individual[1])):
        if random.random() < rot_indpb:
            individual[1][i] = random.randint(0, 5)
    return (individual,)


def _box_volume(box: Box) -> int:
    dx, dy, dz = box.dims
    return int(dx * dy * dz)


def _evaluate(
    individual: List[List[int]],
    boxes: Sequence[Box],
    pallet_dims: Tuple[int, int, int],
    skip_penalty: float,
) -> Tuple[float]:
    order, rotations = individual
    ordered_boxes = [boxes[i] for i in order]
    ordered_rotations = [rotations[i] for i in order]

    pallet = Pallet(
        length_mm=pallet_dims[0],
        width_mm=pallet_dims[1],
        height_mm=pallet_dims[2],
    )
    placed, skipped = pack_boxes_with_rotations(ordered_boxes, pallet, ordered_rotations)

    placed_volume = sum(_box_volume(b) for b in placed)
    skipped_volume = sum(_box_volume(b) for b in skipped)
    total_volume = sum(_box_volume(b) for b in boxes)

    pallet_volume = pallet.length_mm * pallet.width_mm * pallet.height_mm
    density = placed_volume / pallet_volume if pallet_volume > 0 else 0.0
    skipped_ratio = skipped_volume / total_volume if total_volume > 0 else 0.0

    fitness = density - skip_penalty * skipped_ratio
    return (fitness,)


def run_evolution(
    boxes: Sequence[Box],
    pallet_dims: Tuple[int, int, int],
) -> Tuple[List[List[int]], tools.Logbook]:
    random.seed(RANDOM_SEED)
    _ensure_creator()

    toolbox = base.Toolbox()
    box_count = len(boxes)
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        partial(_make_individual, box_count),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate",
        _evaluate,
        boxes=boxes,
        pallet_dims=pallet_dims,
        skip_penalty=GA_SKIP_PENALTY,
    )
    toolbox.register("mate", _mate)
    toolbox.register("mutate", _mutate, order_indpb=GA_ORDER_MUTPB, rot_indpb=GA_ROT_MUTPB)
    toolbox.register("select", tools.selTournament, tournsize=GA_TOURN_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", min)
    stats.register("avg", lambda values: float(sum(values)) / len(values) if values else 0.0)
    stats.register("max", max)
    hall = tools.HallOfFame(1)

    processes = GA_POOL_PROCESSES if GA_POOL_PROCESSES > 0 else (os.cpu_count() or 1)

    with mp.Pool(processes=processes) as pool:
        toolbox.register("map", pool.map)
        population = toolbox.population(n=GA_POP_SIZE)
        population, logbook = algorithms.eaSimple(
            population,
            toolbox,
            cxpb=GA_CXPB,
            mutpb=GA_MUTPB,
            ngen=GA_NGEN,
            stats=stats,
            halloffame=hall,
            verbose=GA_VERBOSE,
        )

    return hall[0], logbook


def pack_from_individual(
    individual: List[List[int]],
    boxes: Sequence[Box],
    pallet_dims: Tuple[int, int, int],
) -> Tuple[Pallet, List[Box], List[Box]]:
    order, rotations = individual
    ordered_boxes = [boxes[i] for i in order]
    ordered_rotations = [rotations[i] for i in order]

    pallet = Pallet(
        length_mm=pallet_dims[0],
        width_mm=pallet_dims[1],
        height_mm=pallet_dims[2],
    )
    placed, skipped = pack_boxes_with_rotations(ordered_boxes, pallet, ordered_rotations)
    return pallet, placed, skipped
