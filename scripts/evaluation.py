import argparse
import json
import h5py
import numpy as np
import csv
import imp

from environments import rlgymenv
import gym

import policyopt
from policyopt import SimConfig, rl, util, nn, tqdm
import os.path

def main():
    np.set_printoptions(suppress=True, precision=5, linewidth=1000)

    parser = argparse.ArgumentParser()
    # MDP options
    parser.add_argument('policy', type=str)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--max_traj_len', type=int, default=None) # only used for saving
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--count', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()

    #filenames = os.listdir(args.policy)
    csvf = open(args.policy[:-3]+'.csv', 'w')
    csvwriter = csv.writer(csvf)

    dataf = open(args.policy[:-3]+ 'full.csv', 'w')
    datawriter = csv.writer(dataf)
    #csvwriter.writerow(['filename', 'average', 'std'])

    # Load the saved state
    if args.policy.find('reacher') > 0:
        key_iter = 200
    elif args.policy.find('humanoid') > 0:
        key_iter = 1500
    else:
        key_iter = 500

    policy_file, policy_key = util.split_h5_name(args.policy+'/snapshots/iter%07d'%key_iter)
    print 'Loading policy parameters from %s in %s' % (policy_key, policy_file)
    with h5py.File(policy_file, 'r') as f:
        train_args = json.loads(f.attrs['args'])
        dset = f[policy_key]
        import pprint
        pprint.pprint(dict(dset.attrs))

    if args.policy.find('shared1') > 0:
        sharednet = True
    else:
        sharednet = False
        
    # Initialize the MDP
    env_name = train_args['env_name']
    print 'Loading environment', env_name
    mdp = rlgymenv.RLGymMDP(env_name)
    util.header('MDP observation space, action space sizes: %d, %d\n' % (mdp.obs_space.dim, mdp.action_space.storage_size))

    if args.max_traj_len is None:
        args.max_traj_len = mdp.env_spec.timestep_limit
    util.header('Max traj len is {}'.format(args.max_traj_len))

    # Initialize the policy and load its parameters

    enable_obsnorm = bool(train_args['enable_obsnorm']) if 'enable_obsnorm' in train_args else train_args['obsnorm_mode'] != 'none'
    if isinstance(mdp.action_space, policyopt.ContinuousSpace):
        policy_cfg = rl.GaussianPolicyConfig(
            hidden_spec=train_args['policy_hidden_spec'],
            min_stdev=0.,
            init_logstdev=0.,
            enable_obsnorm=enable_obsnorm)
        policy = rl.GaussianPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GaussianPolicy', use_shared_std_network=sharednet)
    else:
        policy_cfg = rl.GibbsPolicyConfig(
            hidden_spec=train_args['policy_hidden_spec'],
            enable_obsnorm=enable_obsnorm)
        policy = rl.GibbsPolicy(policy_cfg, mdp.obs_space, mdp.action_space, 'GibbsPolicy', use_shared_std_network=sharednet)
    policy.load_h5(policy_file, policy_key)

    n = 50
    print 'Evaluating based on {} trajs'.format(n)

    returns = []
    lengths = []
    sim = mdp.new_sim()

    for i_traj in xrange(n):
        iteration = 0
        sim.reset()
        totalr = 0.
        l = 0
        while not sim.done and iteration < args.max_traj_len:
            a = policy.sample_actions(sim.obs[None,:], bool(args.deterministic))[0][0,:]
            r = sim.step(a)
            totalr += r
            l += 1
            iteration += 1

        print i_traj, n, totalr, iteration
        datawriter.writerow([i_traj, n, totalr, iteration])
        returns.append(totalr)
        lengths.append(l)
    avg, std = np.array(returns).mean(), np.array(returns).std()
    print 'Avg Return: ', avg, 'Std: ', std
    csvwriter.writerow([args.policy, avg, std])
    del policy
    #import IPython; IPython.embed()

    csvf.close()
    dataf.close()

if __name__ == '__main__':
    main()
