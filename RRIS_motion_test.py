import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box
import os


class RRISEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }
    def __init__(self, **kwargs):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        # observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/character/mjcf/SN475_SK_shifted.xml', 10, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

    def step(self, a):
        # posbefore = self.sim.data.qpos.copy()
        self.do_simulation(a, self.frame_skip)
        # posafter = self.sim.data.qpos.copy()
        # print(self.sim.data.qpos)

        alive_bonus = 1.0
        # reward = (posafter - posbefore) / self.dt
        # print(reward)
        reward = alive_bonus
        # reward -= 1e-3 * np.square(a).sum()
        # terminated = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        
        # if self.render_mode == "human":
            # self.render()

        return ob, reward, False, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        )
        return self._get_obs()

    # def viewer_setup(self):
        # assert self.viewer is not None
        # self.viewer.cam.trackbodyid = 2
        # self.viewer.cam.distance = self.model.stat.extent * 0.5
        # self.viewer.cam.lookat[2] = 1.15
        # self.viewer.cam.elevation = -20

# if __name__ == "__main__":
#     env = RRISEnv()
#     env.reset()
#     for _ in range(10):
#         # env.render()
#         env.step(env.action_space.sample())
#         print(env.observation_space.shape[0])
#     env.close()