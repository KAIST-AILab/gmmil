python scripts/imitate_mj.py --mode ga \
  --env Walker2d-v1 \
  --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 \
  --limit_trajs 11 \
  --data_subsamp_freq 20 \
  --favor_zero_expert_reward 0 \
  --min_total_sa 50000 \
  --max_iter 1001 \
  --reward_include_time 0 \
  --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=walker,num_trajs=11,run=0.h5 \

  #--reward_type l2ball \
  #python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 11 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --                min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=walker,num_trajs=11,run=0.h5
  #python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 18 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --                min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=walker,num_trajs=18,run=0.h5
  # python scripts/imitate_mj.py --mode ga --env Walker2d-v1 --data imitation_runs/modern_stochastic/trajs/trajs_walker.h5 --limit_trajs 25 --data_subsamp_freq 20 --favor_zero_expert_reward 0 --                min_total_sa 50000 --max_iter 501 --reward_type l2ball --reward_include_time 0 --log imitation_runs/modern_stochastic/checkpoints_all/alg=fem,task=walker,num_trajs=25,run=0.h5

