from SwarmSim.SingleSwarmSim import *
from SwarmSim.ExpHelper import *
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")

LOG_DIR = "exp_run_results/"
LOG_FN = "full_run_01.txt"
LOG_STATS = "full_run_01_stats.txt"
LOG_DIFF_STATS="full_run_01_diff_stats.txt"
log_f = open(LOG_DIR + LOG_FN, 'w')
log_f_stats = open(LOG_DIR + LOG_STATS,'w')
log_f_diff_stats = open(LOG_DIR + LOG_DIFF_STATS,'w')

NUM_RUNS = 30
NUM_TRAINING_STEPS = 100
NUM_INFERENCE_STEPS = 50

# Hardcoding the location of the gusts.
WIND_WINDOWS = [[25, 50], [110, 135]]

SWARM_SIZE = 9
SWARM_TYPE = "planar"
START = [0, 0, 0]
END = [2, 2, 2]  # [100,100,100]
ANIMATE = False

swarm_options = [SWARM_SIZE, SWARM_TYPE, "b", START, END]

log_f.write("n: {}, type: {}, runs: {}, n_train: {}, n_inf: {}, path: {} to {}\n\n".format(SWARM_SIZE,
                                                                                           SWARM_TYPE,
                                                                                           NUM_RUNS,
                                                                                           NUM_TRAINING_STEPS,
                                                                                           NUM_INFERENCE_STEPS,
                                                                                           np.array(START),
                                                                                           np.array(END)))
log_f.write("run, seed, "
            "dr mse, dr rmse, dr_r2_score, "
            "model unstructured mse, model unstructured rmse, model unstructured r2_score, "
            "model structured mse, model_structured rmse, model structured r2_score\n")

log_f_stats.write("DR MSE Mean, DR MSE std, Unstructured MSE Mean, Unstructured MSE std, Structured MSE Mean, Structured MSE Std, "
				  "DR RMSE Mean, DR RMSE std, Unstructured RMSE Mean, Unstructured RMSE std, Structured RMSE Mean, Structured RMSE Std, "
				  "DR R2 Mean, DR R2 std, Unstructured R2 Mean, Unstructured R2 std, Structured R2 Mean, Structured R2 Std\n")
log_f_diff_stats.write("MSE (unstruct - dr) mean, MSE (unstruct - dr) std, MSE (struct - dr) mean, MSE (struct - dr) std, MSE (struct - unstruct) mean, MSE (struct - unstruct) std, "
					   "RMSE (unstruct - dr) mean, RMSE (unstruct - dr) std, RMSE (struct - dr) mean, RMSE (struct - dr) std, RMSE (struct - unstruct) mean, RMSE (struct - unstruct) std, "
					   "R2 (unstruct - dr) mean, R2 (unstruct - dr) std, R2 (struct - dr) mean, R2 (struct - dr) std, R2 (struct - unstruct) mean, R2 (struct - unstruct) std\n")

error_mat = np.zeros((NUM_RUNS, 9, 3)) # 3 coordinates for each we have 3 measures (mse, rmse, r2) = 9; 3 models
error_diffs = np.zeros((NUM_RUNS, 9, 3)) # All differences between the outputs
for n in range(NUM_RUNS):
    print('run:', n + 1)

    # rnd_seed = 0 # when set to 0 we don't get the LINALG exception
    rnd_seed = random.randint(0, 10000000)
    # rnd_seed = 1028791
    print('rnd_seed:', rnd_seed)

    # =========================================================================

    # baseline
    sim = SingleSwarmSim(swarm_options, rnd_seed, NUM_TRAINING_STEPS, NUM_INFERENCE_STEPS, None, 'Ground truth', WIND_WINDOWS, ANIMATE)

    for i in range(NUM_TRAINING_STEPS + NUM_INFERENCE_STEPS):
        sim.tick()

    print('Ground truth trajectories: generated!\n')

    # =========================================================================

    # without Model
    sim = SingleSwarmSim(swarm_options, rnd_seed, NUM_TRAINING_STEPS, NUM_INFERENCE_STEPS, None, 'Dead reckoning', WIND_WINDOWS, ANIMATE)
    for i in range(NUM_TRAINING_STEPS):
        sim.tick()

    # starting inference with FALSE means we DONT use the model
    # only dead reckoning
    sim.start_inference(False)
    dr_mse_accumulated = np.zeros((3,), dtype=float)
    dr_rmse_accumulated = np.zeros((3,), dtype=float)
    dr_r2_score_accumulated = np.zeros((3,), dtype=float)
    for i in range(NUM_INFERENCE_STEPS):
        sim.tick()

        target_locations = sim.dump_drone_locations(True)
        dr_locations = sim.dump_drone_locations(False)
        mse_current = calc_mse(target_locations, dr_locations)
        dr_mse_accumulated += mse_current
        dr_rmse_accumulated += np.sqrt(mse_current)
        dr_r2_score_accumulated += calc_r2_score(target_locations, dr_locations)

    dr_mse_avg = dr_mse_accumulated / NUM_INFERENCE_STEPS
    dr_rmse_avg = dr_rmse_accumulated / NUM_INFERENCE_STEPS
    dr_r2_score_avg = dr_r2_score_accumulated / NUM_INFERENCE_STEPS

    print('DR:\nMSE AVG: {}; RMSE AVG: {}; R2 SCORE AVG: {}'.format(dr_mse_avg, dr_rmse_avg, dr_r2_score_avg))
    print('\n\n')

    # =========================================================================

    # with Model
    sim = SingleSwarmSim(swarm_options, rnd_seed, NUM_TRAINING_STEPS, NUM_INFERENCE_STEPS, False, 'Unstructured regressor', WIND_WINDOWS, ANIMATE)
    for i in range(NUM_TRAINING_STEPS):
        sim.tick()

    # starting inference with TRUE means we use the model
    sim.start_inference(True)
    model_mse_accumulated = np.zeros((3,), dtype=float)
    model_rmse_accumulated = np.zeros((3,), dtype=float)
    model_r2_score_accumulated = np.zeros((3,), dtype=float)
    for i in range(NUM_INFERENCE_STEPS):
        sim.tick()

        target_locations = sim.dump_drone_locations(True)

        model_locations = sim.dump_drone_locations(False)
        mse_current = calc_mse(target_locations, model_locations)
        model_mse_accumulated += mse_current
        model_rmse_accumulated += np.sqrt(mse_current)
        model_r2_score_accumulated += calc_r2_score(target_locations, model_locations)
    model_unstructured_mse_avg = model_mse_accumulated / NUM_INFERENCE_STEPS
    model_unstructured_rmse_avg = model_rmse_accumulated / NUM_INFERENCE_STEPS
    model_unstructured_r2_score_avg = model_r2_score_accumulated / NUM_INFERENCE_STEPS

    print('Unstructured model:\nMSE AVG: {}; RMSE AVG: {}; R2 SCORE AVG: {}'.format(model_unstructured_mse_avg,
                                                                                    model_unstructured_rmse_avg,
                                                                                    model_unstructured_r2_score_avg))
    print('\n\n')

    # =========================================================================

    # with Model
    sim = SingleSwarmSim(swarm_options, rnd_seed, NUM_TRAINING_STEPS, NUM_INFERENCE_STEPS, True, 'Temporal GCRF', WIND_WINDOWS, ANIMATE)
    for i in range(NUM_TRAINING_STEPS):
        sim.tick()

    # starting inference with TRUE means we use the model
    sim.start_inference(True)
    model_mse_accumulated = np.zeros((3,), dtype=float)
    model_rmse_accumulated = np.zeros((3,), dtype=float)
    model_r2_score_accumulated = np.zeros((3,), dtype=float)
    for i in range(NUM_INFERENCE_STEPS):
        sim.tick()

        target_locations = sim.dump_drone_locations(True)

        model_locations = sim.dump_drone_locations(False)
        mse_current = calc_mse(target_locations, model_locations)
        model_mse_accumulated += mse_current
        model_rmse_accumulated += np.sqrt(mse_current)
        model_r2_score_accumulated += calc_r2_score(target_locations, model_locations)
    model_structured_mse_avg = model_mse_accumulated / NUM_INFERENCE_STEPS
    model_structured_rmse_avg = model_rmse_accumulated / NUM_INFERENCE_STEPS
    model_structured_r2_score_avg = model_r2_score_accumulated / NUM_INFERENCE_STEPS

    print(
        'GCRF:\nMSE AVG: {}; RMSE AVG: {}; R2 SCORE AVG: {}'.format(model_structured_mse_avg, model_structured_rmse_avg,
                                                                    model_structured_r2_score_avg))
    print('\n\n')

    error_mat[n, 0:3, 0] = dr_mse_avg # dr mse
    error_mat[n, 3:6, 0] = dr_rmse_avg  # dr rmse
    error_mat[n, 6:9, 0] = dr_r2_score_avg  # dr r2
    error_mat[n, 0:3, 1] = model_unstructured_mse_avg # unstructured
    error_mat[n, 3:6, 1] = model_unstructured_rmse_avg
    error_mat[n, 6:9, 1] = model_unstructured_r2_score_avg
    error_mat[n, 0:3, 2] = model_structured_mse_avg # structured
    error_mat[n, 3:6, 2] = model_structured_mse_avg  # structured
    error_mat[n, 6:9, 2] = model_structured_mse_avg  # structured

    error_diffs[n, 0:3, 0] = (model_unstructured_mse_avg - dr_mse_avg) # Unstructured - DR
    error_diffs[n, 3:6, 0] = (model_unstructured_rmse_avg - dr_rmse_avg)
    error_diffs[n, 6:9, 0] = (model_unstructured_r2_score_avg - dr_r2_score_avg)
    error_diffs[n, 0:3, 1] = (model_structured_mse_avg - dr_mse_avg) # Structured - DR
    error_diffs[n, 3:6, 1] = (model_structured_rmse_avg - dr_rmse_avg)
    error_diffs[n, 6:9, 1] = (model_structured_r2_score_avg - dr_r2_score_avg)
    error_diffs[n, 0:3, 2] = (model_structured_mse_avg - model_unstructured_mse_avg) # Structured - Unstructured
    error_diffs[n, 3:6, 2] = (model_structured_rmse_avg - model_unstructured_rmse_avg)
    error_diffs[n, 6:9, 2] = (model_structured_r2_score_avg - model_unstructured_r2_score_avg)

    log_f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(n + 1, rnd_seed, dr_mse_avg, dr_rmse_avg, dr_r2_score_avg, model_unstructured_mse_avg, model_unstructured_rmse_avg, model_unstructured_r2_score_avg, model_structured_mse_avg, model_structured_rmse_avg, model_structured_r2_score_avg))

error_means=np.mean(error_mat, axis=0)
error_diffs_means=np.mean(error_diffs,axis=0)

error_stds=np.std(error_mat, axis=0)
error_diffs_stds=np.std(error_diffs,axis=0)

log_f_stats.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(error_means[0:3,0], error_stds[0:3,0], error_means[0:3,1], error_stds[0:3,1], error_means[0:3,2], error_stds[0:3,2],
																								  error_means[3:6,0], error_stds[3:6,0], error_means[3:6,1], error_stds[3:6,1], error_means[3:6,2], error_stds[3:6,1],
																								  error_means[6:9,0], error_stds[6:9,0], error_means[6:9,1], error_stds[6:9,1], error_means[6:9,2], error_stds[6:9,2]))

log_f_diff_stats.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(error_diffs_means[0:3,0], error_diffs_stds[0:3,0], error_diffs_means[0:3,1], error_diffs_stds[0:3,1], error_diffs_means[0:3,2], error_diffs_stds[0:3,2],
																								       error_diffs_means[3:6,0], error_diffs_stds[3:6,0], error_diffs_means[3:6,1], error_diffs_stds[3:6,1], error_diffs_means[3:6,2], error_diffs_stds[3:6,1],
																								       error_diffs_means[6:9,0], error_diffs_stds[6:9,0], error_diffs_means[6:9,1], error_diffs_stds[6:9,1], error_diffs_means[6:9,2], error_diffs_stds[6:9,2]))


log_f.close()
print('Done!')
