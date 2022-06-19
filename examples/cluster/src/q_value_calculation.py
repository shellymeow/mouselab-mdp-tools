import sys
from pathlib import Path

import dill as pickle

from mouselab.env_utils import get_ground_truths_from_json
from mouselab.exact_utils import timed_solve_env
from mouselab.mouselab import MouselabEnv

experiment_setting = sys.argv[1]
ground_truth_file = sys.argv[2]

# make folder we need
Path(__file__).resolve().parents[1].joinpath("output").mkdir(
    parents=True, exist_ok=True
)

states = get_ground_truths_from_json(ground_truth_file)
env_increasing = MouselabEnv.new_symmetric_registered(experiment_setting)
q, v, pi, info = timed_solve_env(env_increasing, save_q=True, ground_truths=states)

path = (
    Path(__file__)
    .resolve()
    .parents[1]
    .joinpath(f"output/example_q_dict_{experiment_setting}.pickle")
)
with open(path, "wb") as f:
    pickle.dump(info, f)
