
# Generative Moment Matching Imitation Learning

## Kee-Eung Kim and Hyun Soo Park
This code is used for the paper [Imitation Learning via Kernel Mean Embedding](http://ailab.kaist.ac.kr/papers/KP2018)
The implementation is based on [Jonathan Ho's GAIL (Generative Adversarial Imitation Learning) code](https://github.com/openai/imitation).

Contains an implementation of Trust Region Policy Optimization (Schulman et al., 2015) and Generative Adversarial Imitation Learning (Jonathan et al., 2016).

Dependencies:

* Python 2.7
* OpenAI Gym >= 0.1.0, mujoco_py >= 0.4.0
* numpy >= 1.10.4, scipy >= 0.17.0, theano >= 0.8.2
* h5py, pytables, pandas, matplotlib

Provided files:

* ``expert_policies/*`` are the expert policies, trained by TRPO (``scripts/run_rl_mj.py``) on the true costs
* ``scripts/im_pipeline.py`` is the main training and evaluation pipeline. This script is responsible for sampling data from experts to generate training data, running the training code (``scripts/imitate_mj.py``), and evaluating the resulting policies.
* ``pipelines/*`` are the experiment specifications provided to ``scripts/im_pipeline.py``
* ``results/*`` contain evaluation data for the learned policies
