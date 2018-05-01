from . import nn, rl, util, RaggedArray, ContinuousSpace, FiniteSpace, optim, thutil
import numpy as np
from contextlib import contextmanager
import theano; from theano import tensor

from scipy.optimize import fmin_l_bfgs_b

class MMDReward(object):
    # This is just copy version of LinearReward
    # TODO-LIST : reward function equals to MMD witness function!!
    # Consider only Gaussian Kernel first
    # TODO-0 : Start from the Linear Kernel!
    # TODO-1 : Where we apply the median heuristic to determine bandwidth parameters?
    # TODO-2 : How can we implement Gaussian Kernel function?

    def __init__(self,
            obsfeat_space, action_space,
            #mode,
                 #  In LinearReward : l2ball, simplex
                 #  In this case, we consider only MMDReward
            enable_inputnorm, favor_zero_expert_reward,
            include_time,
            time_scale,
            exobs_Bex_Do, exa_Bex_Da, ext_Bex,
            # sqscale=.01,
            # quadratic_features=False
            kernel_bandwidth_params,
            kernel_batchsize
            ):

        self.obsfeat_space, self.action_space = obsfeat_space, action_space
        #assert mode in ['l2ball', 'simplex']
        #print 'Linear reward function type: {}'.format(mode)
        #self.simplex = mode == 'simplex'
        self.favor_zero_expert_reward = favor_zero_expert_reward
        self.include_time = include_time
        self.time_scale = time_scale
        #self.sqscale = sqscale
        #self.quadratic_features = quadratic_features
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
        self.expert_feat_Df = self._compute_featexp(self.exobs_Bex_Do, self.exa_Bex_Da, self.ext_Bex)
        # The current reward function
        # feat_dim = self.expert_feat_Df.shape[0]
        #print 'Linear reward: {} features'.format(feat_dim)
        #if self.simplex:
        #    # widx is the index of the most discriminative reward function
        #    self.widx = np.random.randint(feat_dim)
        #else:
        # w is a weight vector ; FEM uses weights
        # self.w = np.random.randn(feat_dim)
        # self.w /= np.linalg.norm(self.w) + 1e-8
        self.reward_bound = 0.

        # Arguments for MMD Reward
        self.kernel_bandwidth_params = kernel_bandwidth_params,
        self.kernel_batchsize = kernel_batchsize

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
            if self.quadratic_features:
                feat_cols = [obsfeat_B_Do, a_B_Da]
                if self.include_time:
                    feat_cols.extend([scaledt_B_1])
                feat = np.concatenate(feat_cols, axis=1)
                quadfeat = (feat[:,:,None] * feat[:,None,:]).reshape((B,-1))
                feat_B_Df = np.concatenate([feat,quadfeat,np.ones((B,1))], axis=1)
            else:
                feat_cols = [obsfeat_B_Do, a_B_Da, (self.sqscale*obsfeat_B_Do)**2, (self.sqscale*a_B_Da)**2]
                if self.include_time:
                    feat_cols.extend([scaledt_B_1, scaledt_B_1**2, scaledt_B_1**3])
                feat_cols.append(np.ones((B,1)))
                feat_B_Df = np.concatenate(feat_cols, axis=1)

        else:
            assert not self.quadratic_features
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

        if self.simplex:
            feat_B_Df = np.concatenate([feat_B_Df, -feat_B_Df], axis=1)

        assert feat_B_Df.ndim == 2 and feat_B_Df.shape[0] == B
        return feat_B_Df


    def _compute_featexp(self, obsfeat_B_Do, a_B_Da, t_B):
        return self._featurize(obsfeat_B_Do, a_B_Da, t_B).mean(axis=0)


    def fit(self, obsfeat_B_Do, a_B_Da, t_B, _unused_exobs_Bex_Do, _unused_exa_Bex_Da, _unused_ext_Bex):
        # Ignore expert data inputs here, we'll use the one provided in the constructor.
        # Current feature expectations
        curr_feat_Df = self._compute_featexp(obsfeat_B_Do, a_B_Da, t_B)

        # Compute adversary reward
        self.w = self.expert_feat_Df - curr_feat_Df
        l2 = np.linalg.norm(self.w)
        self.w /= l2 + 1e-8
        return [('l2', l2, float)]
        #TODO: Is there anything we have to 'fit' with MMD Reward?

    def compute_reward(self, obsfeat_B_Do, a_B_Da, t_B):
        feat_B_Df = self._featurize(obsfeat_B_Do, a_B_Da, t_B)
        r_B = ( feat_B_Df.dot(self.w)) / float(feat_B_Df.shape[1] )
        assert r_B.shape == (obsfeat_B_Do.shape[0],)

        if self.favor_zero_expert_reward:
            self.reward_bound = max(self.reward_bound, r_B.max())
        else:
            self.reward_bound = min(self.reward_bound, r_B.min())
        shifted_r_B = r_B - self.reward_bound
        if self.favor_zero_expert_reward:
            assert (shifted_r_B <= 0).all()
        else:
            assert (shifted_r_B >= 0).all()

        return shifted_r_B

    def update_inputnorm(self, obs_B_Do, a_B_Da):
        if isinstance(self.action_space, ContinuousSpace):
            self.inputnorm.update(np.concatenate([obs_B_Do, a_B_Da], axis=1))
        else:
            self.inputnorm.update(obs_B_Do)
        self.inputnorm_updated = True