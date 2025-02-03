from typing import List, Tuple, Set
import math
import time
import random
from scipy.spatial import distance


def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_candidate_starts(remaining_stitches: set) -> List[Tuple[int, int]]:
    return sorted(
        {start for start, end in remaining_stitches},
        key=lambda pinhole: (pinhole[0] + pinhole[1], pinhole[0], pinhole[1])
    )


def get_legal_next_stitches(temp_remaining: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
                            temp_used_stitches: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
                            previous_end: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    legal_stitches = []
    for stitch in temp_remaining:
        for direction in [stitch, (stitch[1], stitch[0])]:
            if direction in temp_used_stitches or (previous_end and direction[0] == previous_end):
                continue
            legal_stitches.append(direction)
    return legal_stitches


def find_best_next_stitch(legal_stitches: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                          current_position: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    best_stitch = None
    best_distance = float('inf')

    for stitch in legal_stitches:
        direct_distance = distance.euclidean(current_position, stitch[0])
        if direct_distance < best_distance:
            best_distance = direct_distance
            best_stitch = stitch

    return best_stitch


def simulated_annealing(path: List[Tuple[int, Tuple[int, int], Tuple[int, int], float]], initial_temp: float,
                        cooling_rate: float) -> List[Tuple[int, Tuple[int, int], Tuple[int, int], float]]:
    current_path = path[:]
    best_path = path[:]
    best_cost = sum(step[3] for step in path)
    temp = initial_temp

    while temp > 1:
        i, j = sorted(random.sample(range(len(path)), 2))
        new_path = current_path[:]
        new_path[i], new_path[j] = new_path[j], new_path[i]

        new_cost = sum(step[3] for step in new_path)

        if new_cost < best_cost or random.random() < math.exp((best_cost - new_cost) / temp):
            current_path = new_path
            best_cost = new_cost
            best_path = new_path[:]

        temp *= cooling_rate

    return best_path


def find_optimal_path(boxes: List[Tuple[int, int]]) -> Tuple[
    List[Tuple[int, Tuple[int, int], Tuple[int, int], float]], float, float]:
    best_stitch_distance = 0
    remaining_stitches = set()
    best_path = []
    min_transition_distance = float('inf')

    for x, y in boxes:
        stitch1 = ((x, y), (x + 1, y + 1))
        stitch2 = ((x + 1, y), (x, y + 1))
        remaining_stitches.add(stitch1)
        remaining_stitches.add(stitch2)

    candidate_starts = get_candidate_starts(remaining_stitches)

    for start_position in candidate_starts:
        temp_remaining = remaining_stitches.copy()
        temp_used_stitches = set()
        temp_path = []
        temp_transition_distance = 0.0
        temp_stitch_distance = 0.0
        current_position = start_position
        previous_end = None
        temp_step_number = 1

        while temp_remaining:
            legal_stitches = get_legal_next_stitches(temp_remaining, temp_used_stitches, previous_end)
            best_stitch = find_best_next_stitch(legal_stitches, current_position)
            if best_stitch is None:
                break

            temp_remaining.remove(best_stitch if best_stitch in temp_remaining else (best_stitch[1], best_stitch[0]))
            start, end = best_stitch

            transition_distance = calculate_distance(previous_end, start) if previous_end else 0.0
            stitch_distance = calculate_distance(start, end)
            temp_transition_distance += transition_distance
            temp_stitch_distance += stitch_distance

            temp_path.append((temp_step_number, start, end, transition_distance))
            temp_step_number += 1

            temp_used_stitches.add(best_stitch)
            temp_used_stitches.add((best_stitch[1], best_stitch[0]))
            current_position = end
            previous_end = end

        if temp_transition_distance < min_transition_distance:
            min_transition_distance = temp_transition_distance
            best_path = temp_path
            best_stitch_distance = temp_stitch_distance

    best_path = sorted(simulated_annealing(best_path, initial_temp=1000, cooling_rate=0.995), key=lambda step: step[0])
    total_distance = min_transition_distance + best_stitch_distance
    return best_path, min_transition_distance, total_distance


def main():
    design = [
        (7, 18), (8, 18), (6, 19), (7, 19), (8, 19), (7, 20), (8, 20), (9, 20),
        (10, 18), (11, 18), (10, 19), (11, 19), (12, 19), (13, 19), (14, 19),
        (13, 20), (14, 20), (13, 18), (14, 18), (15, 18), (16, 18), (16, 19),
        (17, 19), (16, 20), (19, 18)
    ]

    start_time = time.time()
    optimal_path_test, total_transition_distance, total_distance = find_optimal_path(design)
    end_time = time.time()

    for step in optimal_path_test:
        print(
            f"Step {step[0]:02d}: Stitch from ({step[1][0]:02d}, {step[1][1]:02d}) to ({step[2][0]:02d}, {step[2][1]:02d}), Transition Distance: {step[3]:.2f}")
    print(f"Total transition distance traveled: {total_transition_distance:.2f}")
    print(f"Total distance traveled (including stitches): {total_distance:.2f}")
    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
