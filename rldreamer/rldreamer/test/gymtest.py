from rldreamer.bin import *
import gymnasium as gym

from rldreamer.bin.holographic_transformer import HolographicTransformer

env = gym.make('CartPole-v0')
holographic_transformer = HolographicTransformer(env)