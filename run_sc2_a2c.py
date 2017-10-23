import os
import sys

sys.path.append(os.getcwd())
from functools import partial
from pysc2.env.sc2_env import SC2Env
from baselines.a2c.a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from sc2.policy import FullyConvPolicy
from sc2.sc2toatari import SC2AtariEnv
from absl import flags
from absl.flags import FLAGS


def make_sc2env(id=0, **kwargs):
    env = SC2Env(**kwargs)
    return SC2AtariEnv(env, id=id, dim=FLAGS.resolution)


def train():
    env_args = dict(
        map_name=FLAGS.map_name,
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=0,
        screen_size_px=(FLAGS.resolution,) * 2,
        minimap_size_px=(FLAGS.resolution,) * 2,
        visualize=FLAGS.visualize
    )

    envs = SubprocVecEnv([partial(make_sc2env, id=i, **env_args) for i in range(FLAGS.n_envs)])
    policy_fn = FullyConvPolicy
    try:
        learn(
            policy_fn,
            envs,
            seed=1,
            total_timesteps=int(1e6) * FLAGS.frames,
            lrschedule=FLAGS.lrschedule,
            nstack=FLAGS.frame_stack,
            ent_coef=FLAGS.entropy_weight,
            vf_coef=FLAGS.value_weight,
            max_grad_norm=1.0,
            lr=FLAGS.learning_rate
        )
    except KeyboardInterrupt:
        pass

    envs.close()


def main():
    flags.DEFINE_string("map_name", "MoveToBeacon", "Name of the map")
    flags.DEFINE_integer("frames", 40, "Number of frames in millions")
    flags.DEFINE_integer("step_mul", 8, "sc2 step multiplier")
    flags.DEFINE_integer("n_envs", 1, "Number of sc2 environments to run in parallel")
    flags.DEFINE_integer("resolution", 32, "sc2 resolution")
    flags.DEFINE_integer("frame_stack", 2,
        "atari style frame stacking, need to be at least 2 for baselines a2c to work")
    flags.DEFINE_string("lrschedule", "constant",
        "linear or constant, learning rate schedule for baselines a2c")
    flags.DEFINE_float("learning_rate", 3e-4, "learning rate")
    flags.DEFINE_boolean("visualize", False, "show pygame visualisation")
    flags.DEFINE_float("value_weight", 1.0, "value function loss weight")
    flags.DEFINE_float("entropy_weight", 1e-5, "entropy loss weight")

    FLAGS(sys.argv)

    train()


if __name__ == '__main__':
    main()
