import numpy as np
import gym
from gym.spaces import Discrete, Box
import carla
import gym_carla
import csv

# ================================================================
# Policies
# ================================================================
class DeterministicDiscreteActionLinearPolicy(object):

    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n
        assert len(theta) == (dim_ob + 1) * n_actions
        self.W = theta[0 : dim_ob * n_actions].reshape(dim_ob, n_actions)
        self.b = theta[dim_ob * n_actions : None].reshape(1, n_actions)

    def act(self, ob):
        """
        """
        y = ob.dot(self.W) + self.b
        a = y.argmax()
        return a

class DeterministicContinuousActionLinearPolicy(object):

    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        dim_ac: dimension of action vector
        theta: flat vector of parameters
        """
        self.ac_space = ac_space
        dim_ob = 5#ob_space.shape[0]
        dim_ac = 2#ac_space.shape[0]
        assert len(theta) == (dim_ob + 1) * dim_ac
        self.W = theta[0 : dim_ob * dim_ac].reshape(dim_ob, dim_ac)
        self.b = theta[dim_ob * dim_ac : None]

    def act(self, ob):
        a = ob.dot(self.W) + self.b#np.clip(ob.dot(self.W) + self.b, self.ac_space.low, self.ac_space.high)
        return a

class CEM():
    def __init__(self):
        self.env = gym.make('carla-v0') # Change as needed

    def do_episode(self, policy, discount=1.0, render=False):
        disc_total_rew = 0
        ob = self.env.reset()
        for t in range(self.num_steps):
            a = policy.act(ob)
            (ob, reward, done, _info) = self.env.step(a)
            disc_total_rew += reward * discount**t
            # if render and t%3==0:
            #     env.render()
            # if done: break
        return disc_total_rew

    def noisy_evaluation(self, theta, discount=0.90):
        policy = DeterministicContinuousActionLinearPolicy(theta, 5, 2)
        reward = self.do_episode(policy, discount)
        return reward

    def main(self):
        self.num_steps = 100 # maximum length of episode
        # Alg settings:
        n_iter = 20 # number of iterations of CEM
        batch_size = 4 # number of samples per batch
        elite_frac = 0.25 # fraction of samples used as elite set
        n_elite = int(batch_size * elite_frac)
        extra_std = 0
        extra_decay_time = 10
        dim_theta = (5+1)*2 #(env.observation_space.shape[0]+1) * env.action_space.shape[0]
        f = open("cem_log.csv", "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["itr", "mean", "std"])

        # Initialize mean and standard deviation
        theta_mean = np.array([0, -1, 0, -1, -0.5, 0, 0, 0, 0, 0, 2.5, 0])
        theta_std = 0.1 * np.ones(dim_theta)

        # Now, for the algorithm
        for itr in range(n_iter):
            # Sample parameter vectors
            extra_cov = max(1.0 - itr / extra_decay_time, 0) * extra_std**2
            thetas = np.random.multivariate_normal(mean=theta_mean,
                                                   cov=np.diag(np.array(theta_std**2) + extra_cov),
                                                   size=batch_size)
            # rewards = np.array(map(noisy_evaluation, thetas))
            rewards = np.zeros([batch_size])
            for i in range(batch_size):
                rewards[i] = self.noisy_evaluation(thetas[i], discount=1.0)

            # Get elite parameters
            elite_inds = rewards.argsort()[-n_elite:]
            elite_thetas = thetas[elite_inds]

            # Update theta_mean, theta_std
            theta_mean = elite_thetas.mean(axis=0)
            theta_std = elite_thetas.std(axis=0)
            mean = rewards.mean()
            std = rewards.std()
            print("iteration %i. mean: %8.3g. std: %8.3g"%(itr, mean, std))
            writer.writerow([itr, mean, std])
            self.do_episode(DeterministicContinuousActionLinearPolicy(theta_mean, 5, 2), discount=1.0)
        f.close()

if __name__ == "__main__":
    cem = CEM()
    cem.main()