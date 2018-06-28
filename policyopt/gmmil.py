from . import nn, rl, util, RaggedArray, ContinuousSpace, FiniteSpace, optim, thutil
import policyopt
import numpy as np
import theano.tensor as T
from scipy.spatial.distance import cosine

from contextlib import contextmanager
import theano; from theano import tensor

from scipy.optimize import fmin_l_bfgs_b
import pickle

class MMDReward(object):
    # This is just copy version of LinearReward
    # TODO-LIST : cost function equals to MMD witness function!!
    # Consider only Gaussian Kernel (RBF Kernel)
    # TODO-1 : Determine bandwidth parameters
    # TODO-2 : Implement Radial Basis Kernel function

    def __init__(self,
                 obsfeat_space, action_space,
                 enable_inputnorm, favor_zero_expert_reward,
                 include_time,
                 time_scale,
                 exobs_Bex_Do, exa_Bex_Da, ext_Bex,
                 kernel_bandwidth_params,
                 kernel_batchsize,
                 kernel_reg_weight,
                 use_median_heuristic,
                 use_logscale_reward,
                 save_reward,
                 epsilon
                 ):

        self.obsfeat_space, self.action_space = obsfeat_space, action_space
        self.favor_zero_expert_reward = favor_zero_expert_reward
        self.include_time = include_time
        self.time_scale = time_scale
        self.exobs_Bex_Do, self.exa_Bex_Da, self.ext_Bex = exobs_Bex_Do, exa_Bex_Da, ext_Bex
        self.use_logscale_reward = use_logscale_reward
        self.save_reward = save_reward
        self.epsilon = epsilon

        with nn.variable_scope('inputnorm'):
            # Standardize both observations and actions if actions are continuous
            # otherwise standardize observations only.
            self.inputnorm = (nn.Standardizer if enable_inputnorm else nn.NoOpStandardizer)(
                (obsfeat_space.dim + action_space.dim) if isinstance(action_space, ContinuousSpace)
                    else obsfeat_space.dim)
            self.inputnorm_updated = False
        self.update_inputnorm(self.exobs_Bex_Do, self.exa_Bex_Da) # pre-standardize with expert data

        # Expert feature expectations
        #self.expert_feat_Df = self._compute_featexp(self.exobs_Bex_Do, self.exa_Bex_Da, self.ext_Bex)
        self.expert_feat_B_Df = self._featurize(self.exobs_Bex_Do, self.exa_Bex_Da, self.ext_Bex)

        # Arguments for MMD Reward
        self.kernel_bandwidth_params = kernel_bandwidth_params
        self.kernel_batchsize = kernel_batchsize
        self.kernel_reg_weight = kernel_reg_weight
        self.use_median_heuristic = use_median_heuristic
        self.mmd_square = 1.

        self.expert_sigmas = []
        self.iteration = 0
        self.YY = None

        self.min_param = 100.0
        self.max_param = 300.0

        # MMD reward function
        # - Use Radial Basis Function Kernel
        #   : k(x,y) = \sum exp(- sigma(i) * ||x-y||^2 )
        # - sigmas : Bandwidth parameters
        x = T.matrix('x')
        y = T.matrix('y')
        sigmas = T.vector('sigmas')
        feat_dim = self.expert_feat_B_Df.shape[1]

        # - dist[i]: ||x[i]-y[i]||^2
        # We should normalize x, y w.r.t its dimension
        # since in large dimension, a small difference between x, y
        # makes large difference in total kernel function value.
        normalized_x = x / feat_dim
        normalized_y = y / feat_dim
        dist_B = ((normalized_x)**2).sum(1).reshape((normalized_x.shape[0], 1)) \
               + ((normalized_y)**2).sum(1).reshape((1, normalized_y.shape[0])) \
               - 2*(normalized_x).dot((normalized_y).T)

        rbf_kernel_sum, _ = theano.scan(fn=lambda sigma, distance: T.exp(-sigma*distance),
                                    outputs_info=None,
                                    sequences=sigmas, non_sequences=dist_B)

        rbf_kernel = rbf_kernel_sum.mean(axis=0)

        if self.kernel_reg_weight > 0.0:
            xynorm = T.outer(normalized_x.norm(2, axis=1), normalized_y.norm(2, axis=1))
            rbf_kernel += self.kernel_reg_weight*((normalized_x).dot(normalized_y.T)) / xynorm

        self.kernel_function = theano.function([x, y, sigmas],
                                               [rbf_kernel],
                                               allow_input_downcast=True)

        # Evaluate k( expert, expert )
        if not (self.use_median_heuristic > 0):
            self.kernel_exex_total = self.kernel_function(self.expert_feat_B_Df,
                                                          self.expert_feat_B_Df,
                                                          self.kernel_bandwidth_params)
            self.kernel_exex_total = np.mean(self.kernel_exex_total)

    def _featurize(self, obsfeat_B_Do, a_B_Da, t_B):
        assert self.inputnorm_updated
        assert obsfeat_B_Do.shape[0] == a_B_Da.shape[0] == t_B.shape[0]
        B = obsfeat_B_Do.shape[0]

        # Standardize observations and actions
        if isinstance(self.action_space, ContinuousSpace):
            trans_B_Doa = self.inputnorm.standardize(np.concatenate([obsfeat_B_Do, a_B_Da], axis=1))
            obsfeat_B_Do, a_B_Da = trans_B_Doa[:,:obsfeat_B_Do.shape[1]], trans_B_Doa[:,obsfeat_B_Do.shape[1]:]
            assert obsfeat_B_Do.shape[1] == self.obsfeat_space.dim and a_B_Da.shape[1] == self.action_space.dim

        else:
            assert a_B_Da.shape[1] == 1 and np.allclose(a_B_Da, a_B_Da.astype(int)), 'actions must all be ints'
            obsfeat_B_Do = self.inputnorm.standardize(obsfeat_B_Do)

        # Concatenate with other stuff to get final features
        scaledt_B_1 = t_B[:,None]*self.time_scale
        if isinstance(self.action_space, ContinuousSpace):
            feat_cols = [obsfeat_B_Do, a_B_Da]
            if self.include_time:
                feat_cols.extend([scaledt_B_1, scaledt_B_1**2, scaledt_B_1**3])
            feat_cols.append(np.ones((B,1)))
            feat_B_Df = np.concatenate(feat_cols, axis=1)

        else:
            # Observation-only features
            obsonly_feat_cols = [obsfeat_B_Do, (.01*obsfeat_B_Do)**2]
            if self.include_time:
                obsonly_feat_cols.extend([scaledt_B_1, scaledt_B_1**2, scaledt_B_1**3])
            obsonly_feat_B_f = np.concatenate(obsonly_feat_cols, axis=1)

            # To get features that include actions, we'll have blocks of obs-only features,
            # one block for each action.
            assert a_B_Da.shape[1] == 1
            action_inds = [np.flatnonzero(a_B_Da[:,0] == a) for a in xrange(self.action_space.size)]
            assert sum(len(inds) for inds in action_inds) == B
            action_block_size = obsonly_feat_B_f.shape[1]
            # Place obs features into their appropriate blocks
            blocked_feat_B_Dfm1 = np.zeros((obsonly_feat_B_f.shape[0], action_block_size*self.action_space.size))
            for a in range(self.action_space.size):
                blocked_feat_B_Dfm1[action_inds[a],a*action_block_size:(a+1)*action_block_size] = obsonly_feat_B_f[action_inds[a],:]
            assert np.isfinite(blocked_feat_B_Dfm1).all()
            feat_B_Df = np.concatenate([blocked_feat_B_Dfm1, np.ones((B,1))], axis=1)

        assert feat_B_Df.ndim == 2 and feat_B_Df.shape[0] == B
        return feat_B_Df

    def _get_median_bandwidth(self, feat_B_Df):
        print "Calculating bandwidth parameters..."
        sigmas = []

        N = feat_B_Df.shape[0]
        M = self.expert_feat_B_Df.shape[0]

        index = np.random.choice(N, M, replace=(N < M))

        initial_points = feat_B_Df[index, :] / feat_B_Df.shape[1]
        expert_points = self.expert_feat_B_Df / feat_B_Df.shape[1]

        # sigma_2 : median of pairwise squared-l2 distance among
        #           data points from the expert policy
        if len(self.expert_sigmas) == 0:
            self.YY = np.multiply(expert_points, expert_points).sum(axis=1).reshape((1, -1))

            dist_matrix = self.YY + (self.YY).T - 2 * np.matmul(expert_points, expert_points.T)
            dist_array = np.absolute(np.asarray(dist_matrix).reshape(-1))
            sigma_2 = 1. / np.median(dist_array)
            self.expert_sigmas.append(sigma_2)

            if self.use_median_heuristic == 2:
                self.expert_sigmas.append(1. / np.percentile(dist_array, 25))
                self.expert_sigmas.append(1. / np.percentile(dist_array, 75))

        if self.use_median_heuristic < 4:
            sigmas.extend(self.expert_sigmas)

        # sigma_1 : median of pairwise squared-l2 distance between
        #           data points from the expert policy and from initial policy
        XX = np.multiply(initial_points, initial_points).sum(axis=1).reshape((1, -1))

        dist_matrix = XX + (self.YY).T - 2 * np.matmul(initial_points, expert_points.T)
        dist_array = np.absolute(np.asarray(dist_matrix).reshape(-1))
        sigma_1 = 1. / np.median(dist_array)
        sigmas.append(sigma_1)

        # - use_median_heuristic 2 :
        #  also use lower quantile (.25 percentile) and upper quantile (.75)
        if self.use_median_heuristic == 2:
            sigmas.append(1. / np.percentile(dist_array, 25))
            sigmas.append(1. / np.percentile(dist_array, 75))

        print "sigmas : ", sigmas

        return sigmas

    # grid search for bandwidth parameter which maximizes MMD^2
    def _get_bandwidth_with_parameter_search(self, feat_B_Df, min_value, max_value, grid_num):
        print("Executing grid search for bandwidth parameter...")
        params = np.linspace(min_value, max_value, grid_num)
        max_param = 0
        max_mmdsquare = 0

        N = feat_B_Df.shape[0]
        M = self.expert_feat_B_Df.shape[0]

        batchsize = self.kernel_batchsize

        total_index = range(len(feat_B_Df))
        start_index = [index for index in total_index[0:len(feat_B_Df):batchsize]]
        end_index = [index for index in total_index[batchsize:len(feat_B_Df):batchsize]]
        end_index.append(len(feat_B_Df))
        indices_list = [range(start, end) for (start, end) in zip(start_index, end_index)]

        print('parameter,  mmd_square')
        for param in params:
            kernel_learned_total = 0
            kernel_expert_total = 0
            # kernel_exex_total = 0

            for indices in indices_list:
                kernel_learned = \
                    self.kernel_function(feat_B_Df,
                                         feat_B_Df[indices, :],
                                         [param])

                kernel_learned_total += np.sum(np.sum(kernel_learned, axis=0))

                kernel_expert = \
                    self.kernel_function(self.expert_feat_B_Df,
                                         feat_B_Df[indices, :],
                                         [param])
                kernel_expert_total += np.sum(np.sum(kernel_expert, axis=0))

            kernel_exex = \
                self.kernel_function(self.expert_feat_B_Df,
                                     self.expert_feat_B_Df,
                                     [param])
            kernel_exex_total = np.sum(np.sum(kernel_exex, axis=0))

            mmd_square = kernel_learned_total / (N * N) - 2. * kernel_expert_total / (N * M) + kernel_exex_total / (M * M)

            print(param, mmd_square)

            if mmd_square > max_mmdsquare:
                max_mmdsquare = mmd_square
                max_param = param

        return [max_param]



    def fit(self, obsfeat_B_Do, a_B_Da, t_B, _unused_exobs_Bex_Do, _unused_exa_Bex_Da, _unused_ext_Bex):
        # In MMD Reward, we don't need to do anything here
        # Return current mmd square value
        outputs = [('MMD^2', self.mmd_square, float)]

        for i in range(len(self.kernel_bandwidth_params)):
            output = (('sigma')+str(i+1), self.kernel_bandwidth_params[i], float)
            outputs.append(output)

        return outputs

    def compute_reward(self, obsfeat_B_Do, a_B_Da, t_B):
        # Features from Learned Policy Trajectory
        feat_B_Df = self._featurize(obsfeat_B_Do, a_B_Da, t_B)
        # Note thant features from expert trajectory : self.expert_feat_B_Df
        cost_B = np.zeros(feat_B_Df.shape[0])

        N = feat_B_Df.shape[0]
        M = self.expert_feat_B_Df.shape[0]

        if self.use_median_heuristic == 4 and self.iteration % 100 == 0:
            self.kernel_bandwidth_params = \
                self._get_bandwidth_with_parameter_search(
                    feat_B_Df, self.min_param, self.max_param, 100)

            self.min_param = max(self.kernel_bandwidth_params[0] - 100.0, 0.0)
            self.max_param = min(self.kernel_bandwidth_params[0] + 100.0, 1000.0)

            self.kernel_exex_total = self.kernel_function(self.expert_feat_B_Df,
                                                          self.expert_feat_B_Df,
                                                          self.kernel_bandwidth_params)
            self.kernel_exex_total = np.sum(np.sum(self.kernel_exex_total, axis=0))
            self.kernel_exex_total /= (M * M)

        if len(self.kernel_bandwidth_params) == 0 or \
           (self.use_median_heuristic == 3 and self.iteration % 50 == 0):

            self.kernel_bandwidth_params = self._get_median_bandwidth(feat_B_Df)
            self.kernel_exex_total = self.kernel_function(self.expert_feat_B_Df,
                                                          self.expert_feat_B_Df,
                                                          self.kernel_bandwidth_params)
            self.kernel_exex_total = np.sum(np.sum(self.kernel_exex_total, axis=0))

            self.kernel_exex_total /= (M * M)

        kernel_learned_total = 0
        kernel_expert_total = 0

        batchsize = self.kernel_batchsize

        total_index = range(len(t_B))
        start_index = [index for index in total_index[0:len(t_B):batchsize]]
        end_index = [index for index in total_index[batchsize:len(t_B):batchsize]]
        end_index.append(len(t_B))
        indices_list = [range(start, end) for (start, end) in zip(start_index, end_index)]

        for indices in indices_list:
            kernel_learned = \
                self.kernel_function(feat_B_Df,
                                     feat_B_Df[indices, :],
                                     self.kernel_bandwidth_params)

            kernel_learned_total += np.sum(np.sum(kernel_learned, axis=0))

            kernel_expert = \
                self.kernel_function(self.expert_feat_B_Df,
                                     feat_B_Df[indices, :],
                                     self.kernel_bandwidth_params)
            kernel_expert_total += np.sum(np.sum(kernel_expert, axis=0))

            cost_B[indices] = np.mean(kernel_learned, axis=1) - np.mean(kernel_expert, axis=1)

        # Use unbiased estimator
        # mmd_square = kernel_learned_total / (N * (N - 1)) - 2. * kernel_expert_total / (N * M) + self.kernel_exex_total
        # Use biased estimator
        mmd_square = kernel_learned_total / (N * N) - 2. * kernel_expert_total / (N * M) + self.kernel_exex_total

        if mmd_square > 0:
            self.mmd_square = mmd_square
        else:
            print "(Warning) Estimator for MMD^2 should be positive. Use previous MMD^2 value."

        cost_B /= np.sqrt(self.mmd_square)
        r_B = -cost_B

        reward_max = r_B.max()
        reward_min = r_B.min()
        margin = (reward_max - reward_min) * self.epsilon #0.0001

        if self.favor_zero_expert_reward:
            # 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            if self.use_logscale_reward:
                reward_B = np.log((r_B - reward_min + margin) / (reward_max - reward_min + margin))
            else:
                reward_B = r_B - reward_max

        else:
            # 0 for non-expert-like states, goes to +inf for expert-like states
            # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
            # e.g. walking simulations that get cut off when the robot falls over
            if self.use_logscale_reward:
                reward_B = -np.log((reward_max - r_B + margin) / (reward_max - reward_min + margin))
            else:
                reward_B = r_B - reward_min

        if self.favor_zero_expert_reward:
            assert (reward_B <= 0).all()
        else:
            assert (reward_B >= 0).all()

        self.current_reward = reward_B

        # Save imaginary rewards into pickle file
        # if self.save_reward and self.iteration % 100 == 0:
        #
        #     with open('reward.pk', 'wb') as reward_f:
        #         print("Save imaginary reward into pickle file...")
        #         pickle.dump(self.current_reward, reward_f)

        self.iteration += 1

        return reward_B

    def update_inputnorm(self, obs_B_Do, a_B_Da):
        if isinstance(self.action_space, ContinuousSpace):
            self.inputnorm.update(np.concatenate([obs_B_Do, a_B_Da], axis=1))
        else:
            self.inputnorm.update(obs_B_Do)
        self.inputnorm_updated = True
