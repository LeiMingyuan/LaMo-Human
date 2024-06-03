import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box
import os
import pickle
import time
# from decision_transformer.envs.stablePD import StablePD
# from stablePD import StablePD
#decision_transformer.envs.
class RRISEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # metadata = {
    #     "render_modes": [
    #         "human",
    #         "rgb_array",
    #         "depth_array",
    #     ],
    #     "render_fps": 125,
    # }
    def __init__(self, **kwargs):
        
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.SPD=None
        # observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/../../../character/mjcf/SN475_SK_shifted.xml', 1, **kwargs)
        # self.SPD=StablePD(self.sim)
        utils.EzPickle.__init__(self, **kwargs)
        self.getposition()
        
    def getposition(self):
        dataset_path = "../data/mujoco/RRIS-medium-v2.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)
        index=np.random.randint(0,len(trajectories))
        states=trajectories[index]["observations"]
        states[:,0],states[:,1],states[:,2]=states[:,2].copy(),states[:,1].copy(),states[:,0].copy()
        states[:,27],states[:,28],states[:,29]=states[:,29].copy(),states[:,28].copy(),states[:,27].copy()
        self.init_qpos=states[0][:27]
        self.init_qvel=states[0][27:]


    def step(self, a):
        # print(a)
        # print("init done")
        # posbefore = self.sim.data.qpos.copy()
        # if self.SPD is not None:
        #     action=self.SPD.CalcControlForces(a)
        # else:
        #     action=a
        action1=np.zeros_like(a)
        action=a
        self.set_state(action, np.zeros_like(action1))
        self.do_simulation(action1, self.frame_skip)
        
        
        # posafter = self.sim.data.qpos.copy()
        # print(self.sim.data.qpos)
        #!Pelvis ty is the height of the pelvis
        height= self.sim.data.qpos[1]
        alive_bonus = 1.0
        # reward = (posafter - posbefore) / self.dt
        # print(reward)
        reward = alive_bonus
        # reward -= 1e-3 * np.square(a).sum()
        
        terminated = not (height >-0.3 )
        ob = self._get_obs()
        
        # if self.render_mode == "human":
            # self.render()

        return ob, reward, terminated, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[:], qvel[:]]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos,
            # + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            # + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
    def get_normalized_score(self, x):
        return x/100.0

if __name__ == "__main__":
    env = RRISEnv()
    dataset_path = "../data/mujoco/RRIS-medium-v2.pkl"
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)
    
    # print(env.init_qpos)
    # print(env.init_qvel)
    env.reset()
    i=0
    for i in range(len(trajectories)):
        states=trajectories[i]["observations"]
        states[:,0],states[:,1],states[:,2]=states[:,2].copy(),states[:,1].copy(),states[:,0].copy()
        states[:,27],states[:,28],states[:,29]=states[:,29].copy(),states[:,28].copy(),states[:,27].copy()
        env.init_qpos=states[0,:27]
        env.init_qvel=states[0,27:]
        action=trajectories[i]["next_observations"][:,:27]
        action[:,0],action[:,1],action[:,2]=action[:,2].copy(),action[:,1].copy(),action[:,0].copy()
        # action[:,27],action[:,28],action[:,29]=action[:,29].copy(),action[:,28].copy(),action[:,27].copy()
        env.reset()
        # env.render()
        # action[i],
        for k in range(len(action)):
            env.step(action[k])
            # print("-----------------------------------")
            # print(env.sim.data.ctrl)
            # print(action[i])
            #pause for 5s

            # time.sleep(0.05)
            # print(action)
            env.render(mode='human')
        env.reset()
        
    env.close() 
    