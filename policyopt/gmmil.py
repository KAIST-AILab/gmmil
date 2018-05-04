from . import nn, rl, util, RaggedArray, ContinuousSpace, FiniteSpace, optim, thutil
import policyopt
import numpy as np
import theano.tensor as T

from contextlib import contextmanager
import theano; from theano import tensor

from scipy.optimize import fmin_l_bfgs_b

class MMDReward(object):
    # This is just copy version of LinearReward
    # TODO-LIST : cost function equals to MMD witness function!!
    # Consider only Gaussian Kernel first
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
            use_median_heuristic
            ):

        self.obsfeat_space, self.action_space = obsfeat_space, action_space
        self.favor_zero_expert_reward = favor_zero_expert_reward
        self.include_time = include_time
        self.time_scale = time_scale
        self.exobs_Bex_Do, self.exa_Bex_Da, self.ext_Bex = exobs_Bex_Do, exa_Bex_Da, ext_Bex

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
        self.use_median_heuristic = use_median_heuristic
        self.mmd_square = 1.

        # Reward Bounds
        self.reward_bound = 0.

        # The current reward function
        # TODO: Implement current reward function here
        # Radial Basis Function Kernel : k(x,y) = \sum exp(- sigma(i) * ||x-y||^2 )
        # Bandwidth parameters = sigmas
        x = T.matrix('x') # matrix concatenate x,y
        y = T.matrix('y')
        sigmas = T.vector('sigmas')

        # dist = ||x-y||^2
        dist = (x**2).sum(1).reshape((x.shape[0], 1)) -2*x.dot(y.T) + (y**2).sum(1).reshape((1, y.shape[0]))
        dist = T.clip(dist, 1e-10, 1e20)
        rbf_kernel, _ = theano.scan(fn=lambda sigma, distance: T.exp(-sigma*distance),
                                    outputs_info=None,
                                    sequences=sigmas, non_sequences=dist)

        rbf_kernel_mean = rbf_kernel.mean(axis=0).mean(axis=0)
        rbf_kernel_mean_batchsum = rbf_kernel.mean(axis=0).mean(axis=0).sum()

        self.kernel_function = theano.function([x, y, sigmas],
                                               [rbf_kernel_mean, rbf_kernel_mean_batchsum],
                                               allow_input_downcast=True)

        # Evaluate k( expert, expert )
        if not self.use_median_heuristic:
            _, self.kernel_exex_total = self.kernel_function(self.expert_feat_B_Df,
                                                             self.expert_feat_B_Df,
                                                             self.kernel_bandwidth_params)


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
        print "Calculating bandwitdh parameters..."
        sigmas = []

        N = feat_B_Df.shape[0]
        M = self.expert_feat_B_Df.shape[0]
        index = np.random.choice(N, M, replace=False)
        initial_points = feat_B_Df[index, :]

        # sigma_1 : median of pairwise squared-l2 distance between
        #           data points from the expert policy and from initial policy
        XX = np.multiply(initial_points, initial_points).sum(axis=1)
        YY = np.multiply(self.expert_feat_B_Df, self.expert_feat_B_Df).sum(axis=1)
        dist_matrix = XX + YY.T - 2 * np.matmul(initial_points, self.expert_feat_B_Df.T)
        dist_array = np.asarray(dist_matrix).reshape(-1)
        sigma_1 = 1./np.median(dist_array)
        sigmas.append(sigma_1)

        # sigma_2 : median of pairwise squared-l2 distance among
        #           data points from the expert policy
        dist_matrix = YY + YY.T - 2 * np.matmul(self.expert_feat_B_Df, self.expert_feat_B_Df.T)
        dist_array = np.asarray(dist_matrix).reshape(-1)
        sigma_2 = 1. / np.median(dist_array)
        sigmas.append(sigma_2)

        print "sigmas : ", sigmas

        return sigmas

    def fit(self, obsfeat_B_Do, a_B_Da, t_B, _unused_exobs_Bex_Do, _unused_exa_Bex_Da, _unused_ext_Bex):
        # Ignore expert data inputs here, we'll use the one provided in the constructor.
        # Current feature expectations
        # MMD Reward : Nothing to do

        return [('MMD^2*1000',self.mmd_square*1000, float),
                ('cur_reward',self.current_reward, float)]

    def compute_reward(self, obsfeat_B_Do, a_B_Da, t_B):
        # Features from Learned Policy Trajectory
        feat_B_Df = self._featurize(obsfeat_B_Do, a_B_Da, t_B)
        # Note thant features from expert trajectory : self.expert_feat_B_Df
        cost_B = np.zeros(feat_B_Df.shape[0])

        if len(self.kernel_bandwidth_params) == 0:
            self.kernel_bandwidth_params = self._get_median_bandwidth(feat_B_Df)
            _, self.kernel_exex_total = self.kernel_function(self.expert_feat_B_Df,
                                                             self.expert_feat_B_Df,
                                                             self.kernel_bandwidth_params)

        N = feat_B_Df.shape[0]
        M = self.expert_feat_B_Df.shape[0]

        kernel_learned_total = 0
        kernel_expert_total = 0

        batchsize = self.kernel_batchsize

        total_index = range(len(t_B))
        start_index = [index for index in total_index[0:len(t_B):batchsize]]
        end_index = [index for index in total_index[batchsize:len(t_B):batchsize]]
        end_index.append(len(t_B))
        indices_list = [range(start, end) for (start, end) in zip(start_index, end_index)]

        for indices in indices_list:
            #indices = range(i*batchsize, (i+1)*batchsize)
            kernel_learned, kernel_learned_sum = \
                self.kernel_function(feat_B_Df,
                                     feat_B_Df[indices, :],
                                     self.kernel_bandwidth_params)
            kernel_learned_total += kernel_learned_sum

            kernel_expert, kernel_expert_sum = \
                self.kernel_function(self.expert_feat_B_Df,
                                     feat_B_Df[indices, :],
                                     self.kernel_bandwidth_params)
            kernel_expert_total += kernel_expert_sum

            # Cost function = Unnormalized Witness Function
            cost_B[indices] = kernel_learned - kernel_expert


        print "kernel_learned_total :", kernel_learned_total
        print "kernel_expert_total :", kernel_expert_total
        print "kernel_exex_total : ", self.kernel_exex_total

        self.mmd_square = kernel_learned_total/N - 2.* kernel_expert_total / N + self.kernel_exex_total / M

        print "mmd_square : ", self.mmd_square
        print "mmd : ", np.sqrt(self.mmd_square)

        if self.mmd_square > 0:
            cost_B /= np.sqrt(self.mmd_square)

        r_B = -cost_B

        # TODO: logarithmetic reward?
        if self.favor_zero_expert_reward:
            # 0 for expert-like states, goes to -inf for non-expert-like states
            # compatible with envs with traj cutoffs for good (expert-like) behavior
            # e.g. mountain car, which gets cut off when the car reaches the destination
            # rewards_B = thutil.logsigmoid(scores_B)
            self.reward_bound = max(self.reward_bound, r_B.max())
        else:
            # 0 for non-expert-like states, goes to +inf for expert-like states
            # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
            # e.g. walking simulations that get cut off when the robot falls over
            self.reward_bound = min(self.reward_bound, r_B.min())

        shifted_r_B = r_B - self.reward_bound
        if self.favor_zero_expert_reward:
            assert (shifted_r_B <= 0).all()
        else:
            assert (shifted_r_B >= 0).all()

        self.current_reward = shifted_r_B

        return shifted_r_B

    def update_inputnorm(self, obs_B_Do, a_B_Da):
        if isinstance(self.action_space, ContinuousSpace):
            self.inputnorm.update(np.concatenate([obs_B_Do, a_B_Da], axis=1))
        else:
            self.inputnorm.update(obs_B_Do)
        self.inputnorm_updated = True