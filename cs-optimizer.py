from typing import List, Tuple
import math
from scipy.spatial import distance


def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_optimal_path(boxes: List[Tuple[int, int]]) -> Tuple[
    List[Tuple[int, Tuple[int, int], Tuple[int, int], float]], float]:
    remaining_stitches = set()
    total_distance = float('inf')
    best_path = []

    for x, y in boxes:
        stitch1 = ((x, y), (x + 1, y + 1))
        stitch2 = ((x + 1, y), (x, y + 1))
        remaining_stitches.add(stitch1)
        remaining_stitches.add(stitch2)

    # Evaluate multiple starting positions
    candidate_starts = sorted(
        {start for start, end in remaining_stitches},
        key=lambda pinhole: (pinhole[0] + pinhole[1], pinhole[0], pinhole[1])
    )

    for start_position in candidate_starts:
        temp_remaining = remaining_stitches.copy()
        temp_used_stitches = set()
        temp_path = []
        temp_total_distance = 0.0
        current_position = start_position
        previous_end = None
        temp_step_number = 1

        while temp_remaining:
            best_stitch = None
            best_distance = float('inf')

            for stitch in temp_remaining:
                for direction in [stitch, (stitch[1], stitch[0])]:  # Consider both stitch orientations
                    if direction in temp_used_stitches or (previous_end and direction[0] == previous_end):
                        continue
                    direct_distance = distance.euclidean(current_position, direction[0])
                    total_cost = direct_distance

                    if total_cost < best_distance:
                        best_distance = total_cost
                        best_stitch = direction

            if best_stitch is None:
                break

            temp_remaining.remove(best_stitch if best_stitch in temp_remaining else (best_stitch[1], best_stitch[0]))
            start, end = best_stitch

            transition_distance = calculate_distance(previous_end, start) if previous_end else 0.0
            temp_total_distance += transition_distance
            temp_total_distance += calculate_distance(start, end)

            temp_path.append((temp_step_number, start, end, transition_distance))
            temp_step_number += 1

            temp_used_stitches.add(best_stitch)
            temp_used_stitches.add((best_stitch[1], best_stitch[0]))  # Mark both directions as used
            current_position = end
            previous_end = end

        if temp_total_distance < total_distance:
            total_distance = temp_total_distance
            best_path = temp_path

    return best_path, total_distance


def main():
    design = [
        (7, 18), (8, 18), (6, 19), (7, 19), (8, 19), (7, 20), (8, 20), (9, 20),
        (10, 18), (11, 18), (10, 19), (11, 19), (12, 19), (13, 19), (14, 19),
        (13, 20), (14, 20), (13, 18), (14, 18), (15, 18), (16, 18), (16, 19),
        (17, 19), (16, 20), (19, 18)
    ]

    optimal_path_test, total_distance_test = find_optimal_path(design)

    for step in optimal_path_test:
        print(f"Step {step[0]}: Stitch from {step[1]} to {step[2]}, Transition Distance: {step[3]:.2f}")
    print(f"Total distance traveled: {total_distance_test:.2f}")


if __name__ == "__main__":
    main()
