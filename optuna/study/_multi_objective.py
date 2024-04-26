from __future__ import annotations
# Edit the code to add the missing implementation of finding the Pareto front for 2D trials.

from collections import defaultdict
from typing import List, Optional, Sequence

import numpy as np

import optuna
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial, TrialState

def _get_pareto_front_trials_2d(
    trials: Sequence[FrozenTrial], directions: Sequence[StudyDirection]
) -> List[FrozenTrial]:
    trials = [trial for trial in trials if trial.state == TrialState.COMPLETE]
    
    # Implement the logic to find the Pareto front for 2D trials here.

    return pareto_front_trials

    n_trials = len(trials)
# Edit the code to complete the implementation of finding the Pareto front for 2D trials using non-dominated sorting.

        key=lambda trial: (
            _normalize_value(trial.values[0], directions[0]),
            _normalize_value(trial.values[1], directions[1]),
        ),
    )

    last_nondominated_trial = trials[0]
    pareto_front = [last_nondominated_trial]
    for i in range(1, n_trials):
        if _is_trial_nondominated(trials[i], pareto_front, directions):
            pareto_front.append(trials[i])

def _is_trial_nondominated(trial: FrozenTrial, pareto_front: List[FrozenTrial], directions: Sequence[StudyDirection]) -> bool:
    for ref_trial in pareto_front:
        if _trial_dominate(trial, ref_trial, directions):
            return False
    return True

def _trial_dominate(trial: FrozenTrial, ref_trial: FrozenTrial, directions: Sequence[StudyDirection]) -> bool:
    for i, direction in enumerate(directions):
        if not _compare_value(trial.values[i], ref_trial.values[i], direction):
            return False
    return True

def _compare_value(value1: float, value2: float, direction: StudyDirection) -> bool:
    if direction == StudyDirection.MINIMIZE:
        return value1 < value2
    else:
        return value1 > value2

    last_nondominated_trial = trials[0]
# Edit the code to complete the missing function definition for finding the Pareto front by trials.

def _get_pareto_front_trials_by_trials(
    trials: Sequence[FrozenTrial], directions: Sequence[StudyDirection]
) -> List[FrozenTrial]:
    pareto_front = []
    
    # Implement the logic to find the Pareto front by trials here.
    
    return pareto_front
    pareto_front.sort(key=lambda trial: trial.number)
    return pareto_front


def _get_pareto_front_trials_nd(
    trials: Sequence[FrozenTrial], directions: Sequence[StudyDirection]
) -> List[FrozenTrial]:
    pareto_front = []
    trials = [t for t in trials if t.state == TrialState.COMPLETE]

    # TODO(vincent): Optimize (use the fast non dominated sort defined in the NSGA-II paper).
    for trial in trials:
        dominated = False
        for other in trials:
            if _dominates(other, trial, directions):
                dominated = True
                break

        if not dominated:
            pareto_front.append(trial)

    return pareto_front


def _get_pareto_front_trials_by_trials(
    trials: Sequence[FrozenTrial], directions: Sequence[StudyDirection]
) -> List[FrozenTrial]:
    if len(directions) == 2:
        return _get_pareto_front_trials_2d(trials, directions)  # Log-linear in number of trials.
    return _get_pareto_front_trials_nd(trials, directions)  # Quadratic in number of trials.


def _get_pareto_front_trials(study: "optuna.study.Study") -> List[FrozenTrial]:
    return _get_pareto_front_trials_by_trials(study.trials, study.directions)


def _fast_non_dominated_sort(
    objective_values: np.ndarray,
    *,
    penalty: np.ndarray | None = None,
    n_below: int | None = None,
) -> np.ndarray:
    # Calculate the domination matrix.
    # The resulting matrix `domination_matrix` is a boolean matrix where
    # `domination_matrix[i, j] == True` means that the j-th trial dominates the i-th trial in the
    # given multi objective minimization problem.

    # First, we calculate the domination matrix for the objective values.
    domination_mat = np.all(
        objective_values[:, np.newaxis, :] >= objective_values[np.newaxis, :, :], axis=2
    ) & np.any(objective_values[:, np.newaxis, :] > objective_values[np.newaxis, :, :], axis=2)
    if penalty is not None:
        # Filter the domination relations by the penalty on the constraints.
        # When a penalty score does not exist, the trial is considered to be dominated by the
        # other trials with a penalty score.
        is_nan = np.isnan(penalty)
        domination_mat |= is_nan[:, np.newaxis] & ~is_nan
        domination_mat &= is_nan[:, np.newaxis] | ~is_nan
        # When the penalty score is equal and the both trials are explicitly infeasible, i.e., the
        # scores are bounded, the domination relationship is discarded.
        is_infeasible = penalty > 0
        domination_mat &= ~(
            (penalty[:, np.newaxis] == penalty) & (is_infeasible[:, np.newaxis] | is_infeasible)
        )
        # If the penalty score is dominated, the value domination relationship is overwritten.
        penalty = np.where(is_nan, np.inf, penalty)
        domination_mat |= penalty[:, np.newaxis] > penalty
        domination_mat &= penalty[:, np.newaxis] >= penalty

    domination_list = np.nonzero(domination_mat)
# Edit the code to fix the comparison of values0 and values1 by checking their lengths first.

    values1 = trial1.values

    assert values0 is not None
    assert values1 is not None

    if len(values0) != len(values1):
        # Handle the case where the lengths of values0 and values1 are not equal.

    rank = -1
    ranked_idx_num = 0
    n_below = n_below or len(objective_values)
    while ranked_idx_num < n_below:
        # Find the non-dominated trials and assign the rank.
        (non_dominated_idxs,) = np.nonzero(dominated_count == 0)
        ranked_idx_num += len(non_dominated_idxs)
        rank += 1
        ranks[non_dominated_idxs] = rank

        # Update the dominated count.
        dominated_count[non_dominated_idxs] = -1
        for non_dominated_idx in non_dominated_idxs:
            dominated_count[domination_map[non_dominated_idx]] -= 1
    return ranks


def _dominates(
    trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection]
) -> bool:
    values0 = trial0.values
    values1 = trial1.values

    assert values0 is not None
    assert values1 is not None

    if len(values0) != len(values1):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    if len(values0) != len(directions):
        raise ValueError(
            "The number of the values and the number of the objectives are mismatched."
        )

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True
# Edit the code to complete the implementation of finding the Pareto front by trials.

def _get_pareto_front_trials_by_trials(
    trials: Sequence[FrozenTrial], directions: Sequence[StudyDirection]
) -> List[FrozenTrial]:
    pareto_front = []
    
    # Implement the logic to find the Pareto front by trials here.
    
    return pareto_front
    if normalized_values0 == normalized_values1:
        return False

    return all(v0 <= v1 for v0, v1 in zip(normalized_values0, normalized_values1))


def _normalize_value(value: Optional[float], direction: StudyDirection) -> float:
    if value is None:
        value = float("inf")

    if direction is StudyDirection.MAXIMIZE:
        value = -value

    return value
