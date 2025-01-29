from typing import List, Tuple
import math
from scipy.spatial import distance
import heapq


def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def evaluate_solution(path: List[Tuple[int, int]]) -> Tuple[float, float]:
    total_distance = 0.0
    front_distance = 0.0
    back_distance = 0.0

    for i in range(1, len(path)):
        prev, curr = path[i - 1], path[i]
        dist = calculate_distance(prev, curr)
        total_distance += dist

        if abs(prev[0] - curr[0]) == 1 and abs(prev[1] - curr[1]) == 1:
            front_distance += dist  # Diagonal stitches contribute to front distance
        else:
            back_distance += dist  # Backstitches contribute to back distance

    return total_distance, back_distance


def find_optimal_path_dynamic_start(boxes: List[Tuple[int, int]]) -> Tuple[
    List[Tuple[int, Tuple[int, int], Tuple[int, int]]], float]:
    path = []
    remaining_stitches = set()
    step_number = 1
    total_distance = 0.0
    used_pinhole_positions = set()  # Track used pinholes to avoid undoing stitches

    # Convert boxes into individual stitches
    for x, y in boxes:
        remaining_stitches.add(((x, y), (x + 1, y + 1)))
        remaining_stitches.add(((x + 1, y), (x, y + 1)))

    # Dynamically determine the best starting position based on the first stitch
    current_position = min(
        {start for start, end in remaining_stitches},
        key=lambda pinhole: pinhole[0] + pinhole[1]
    )

    while remaining_stitches:
        # Find the nearest stitch based on shortest travel distance
        next_stitch = min(
            remaining_stitches,
            key=lambda stitch: distance.euclidean(current_position, stitch[0])
        )

        remaining_stitches.remove(next_stitch)
        start, end = next_stitch

        # Ensure the thread does not feed back through the same hole consecutively
        if start in used_pinhole_positions:
            continue

        total_distance += calculate_distance(current_position, start)
        total_distance += calculate_distance(start, end)

        path.append((step_number, start, end))
        step_number += 1

        used_pinhole_positions.add(start)
        used_pinhole_positions.add(end)
        current_position = end

    return path, total_distance


def main():
    design = [
        (7, 18), (8, 18), (6, 19), (7, 19), (8, 19), (7, 20), (8, 20), (9, 20),
        (10, 18), (11, 18), (10, 19), (11, 19), (12, 19), (13, 19), (14, 19),
        (13, 20), (14, 20), (13, 18), (14, 18), (15, 18), (16, 18), (16, 19),
        (17, 19), (16, 20), (19, 18)
    ]

    optimal_path_test, total_distance_test = find_optimal_path_dynamic_start(design)

    # Print the results with step numbers and total distance
    for step in optimal_path_test:
        print(f"Step {step[0]}: Stitch from {step[1]} to {step[2]}")
    print(f"Total distance traveled: {total_distance_test:.2f}")


if __name__ == "__main__":
    main()
