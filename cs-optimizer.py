from typing import List, Tuple
import math
from scipy.spatial import distance
import heapq


def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_optimal_path_dynamic_start(boxes: List[Tuple[int, int]]) -> Tuple[
    List[Tuple[int, Tuple[int, int], Tuple[int, int], float]], float]:
    path = []
    remaining_stitches = set()
    step_number = 1
    total_distance = 0.0
    used_stitches = set()

    for x, y in boxes:
        stitch1 = ((x, y), (x + 1, y + 1))
        stitch2 = ((x + 1, y), (x, y + 1))
        remaining_stitches.add(stitch1)
        remaining_stitches.add(stitch2)

    current_position = min(
        {start for start, end in remaining_stitches},
        key=lambda pinhole: pinhole[0] + pinhole[1]
    )

    previous_end = None
    while remaining_stitches:
        best_stitch = None
        best_distance = float('inf')

        for stitch in remaining_stitches:
            for direction in [stitch, (stitch[1], stitch[0])]:  # Consider both stitch orientations
                if direction in used_stitches or (previous_end and direction[0] == previous_end):
                    continue
                direct_distance = distance.euclidean(current_position, direction[0])
                lookahead_distance = sum(
                    distance.euclidean(direction[1], next_stitch[0])
                    for next_stitch in remaining_stitches if next_stitch != stitch
                ) / max(1, len(remaining_stitches) - 1)

                total_cost = direct_distance + lookahead_distance

                if total_cost < best_distance:
                    best_distance = total_cost
                    best_stitch = direction

        if best_stitch is None:
            break

        remaining_stitches.remove(
            best_stitch if best_stitch in remaining_stitches else (best_stitch[1], best_stitch[0]))
        start, end = best_stitch

        transition_distance = calculate_distance(previous_end, start) if previous_end else 0.0
        total_distance += transition_distance
        total_distance += calculate_distance(start, end)

        path.append((step_number, start, end, transition_distance))
        step_number += 1

        used_stitches.add(best_stitch)
        used_stitches.add((best_stitch[1], best_stitch[0]))  # Mark both directions as used
        current_position = end
        previous_end = end

    return path, total_distance


def main():
    design = [
        (7, 18), (8, 18), (6, 19), (7, 19), (8, 19), (7, 20), (8, 20), (9, 20),
        (10, 18), (11, 18), (10, 19), (11, 19), (12, 19), (13, 19), (14, 19),
        (13, 20), (14, 20), (13, 18), (14, 18), (15, 18), (16, 18), (16, 19),
        (17, 19), (16, 20), (19, 18)
    ]

    optimal_path_test, total_distance_test = find_optimal_path_dynamic_start(design)

    for step in optimal_path_test:
        print(f"Step {step[0]}: Stitch from {step[1]} to {step[2]}, Transition Distance: {step[3]:.2f}")
    print(f"Total distance traveled: {total_distance_test:.2f}")


if __name__ == "__main__":
    main()
