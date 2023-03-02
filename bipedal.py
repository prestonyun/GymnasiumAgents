import gymnasium as gym
import Box2D
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm as tdqm
import matplotlib.pyplot as plt

import dreamerv2.api as dv2

from typing import List

import gymnasium.envs.box2d.bipedal_walker as bp
from gymnasium.error import DependencyNotInstalled

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    ) from e

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0,
) 

LEG_FD = fixtureDef(
    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

LOWER_FD = fixtureDef(
    shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.hull == contact.fixtureA.body
            or self.env.hull == contact.fixtureB.body
        ):
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True

    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False

class bpw(bp.BipedalWalker):
    def __init__(self):
        super().__init__(self)

    def reset(self, seed):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y), fixtures=HULL_FD
        )
        self.hull.color1 = (127, 51, 229)
        self.hull.color2 = (76, 76, 127)
        self.hull.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True
        )

        self.legs: List[Box2D.b2Body] = []
        self.joints: List[Box2D.b2RevoluteJoint] = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LEG_FD,
            )
            leg.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            leg.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=i,
                lowerAngle=-0.8,
                upperAngle=1.1,
            )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LOWER_FD,
            )
            lower.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            lower.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-1.6,
                upperAngle=-0.1,
            )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction

        self.lidar = [LidarCallback() for _ in range(10)]
        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0, 0, 0]))[0], {}

    def step(self, action: np.ndarray):
        assert self.hull is not None

        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1)
            )
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1)
            )
            self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1)
            )
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1)
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            self.joints[0].angle,
            # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0,
        ]
        state += [l.fraction for l in self.lidar]
        assert len(state) == 24

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5

        shaping = (
            130 * pos[0] / SCALE
        )  # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0 * abs(
            state[0]
        )  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        terminated = False
        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True

        if self.render_mode == "human":
            self.render()
        return np.array(state, dtype=np.float32), reward, terminated, False, {}


# Define the neural network model
class BipedalWalkerModel(nn.Module):
    def __init__(self):
        super(BipedalWalkerModel, self).__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the training loop
def train(env, model, num_epochs, batch_size, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    rewards = []
    for epoch in tdqm.tqdm(range(num_epochs)):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        states = []
        actions = []
        rewards_episode = []

        while not done:
            # Get an action from the model
            obs_tensor = torch.from_numpy(np.array(obs)).float().to(device)
            action_tensor = model(obs_tensor)
            action = action_tensor.detach().cpu().numpy()

            # Execute the action in the environment
            obs, reward, done, _, info = env.step(action)

            # Store the state, action, and reward
            states.append(obs_tensor)
            actions.append(action_tensor)
            rewards_episode.append(reward)

            # Update the total reward
            total_reward += reward

        # Compute the discounted rewards
        discounted_rewards = []
        running_reward = 0
        for reward in reversed(rewards_episode):
            running_reward = reward + 0.99 * running_reward
            discounted_rewards.append(running_reward)
        discounted_rewards = list(reversed(discounted_rewards))
        discounted_rewards = torch.tensor(discounted_rewards).float().to(device)

        # Normalize the discounted rewards
        discounted_rewards -= discounted_rewards.mean()
        discounted_rewards /= discounted_rewards.std()

        # Compute the loss and update the model
        optimizer.zero_grad()
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = discounted_rewards[i]
            predicted_action = model(state)
            loss = criterion(predicted_action, action) * reward
            loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1} - Total Reward: {total_reward:.2f}')

        rewards.append(total_reward)

    # Plot the training progress
    plt.plot(rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.show()

def data_collection(num_episodes):
    data = []
    seed = 123
    np.random.seed(seed)
    env = gym.make('BipedalWalker-v3')

    obs_min = env.observation_space.low
    obs_max = env.observation_space.high
    random_state = list(np.random.uniform(obs_min, obs_max))
    print(random_state)

    b = bpw()
    b.reset(seed)
    b.state = random_state
    episode_data = []

    for episode in range(num_episodes):
        obs = b.state
        done = False

        action = env.action_space.sample()
        next_obs, reward, done, _, info = b.step(action)
        episode_data.append((obs, action, reward, next_obs, done))

        random_state = list(np.random.uniform(obs_min, obs_max))

        b = bp.BipedalWalker()
        b.state = random_state
    
    return episode_data


def main():

    env = gym.make('BipedalWalker-v3')

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 1000
    max_episode_length = 1600
    data = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_data = []

        for t in range(max_episode_length):
            action = env.action_space.sample()
            next_obs, reward, done, _, info = env.step(action)
            episode_data.append((obs, action, reward, next_obs, done))
            obs = next_obs

            if done:
                break

        data += episode_data
        print(f'Episode {episode + 1} - Length: {t + 1}')

    obs = np.array([d[0] for d in data])
    actions = np.array([d[1] for d in data])
    rewards = np.array([d[2] for d in data])
    next_obs = np.array([d[3] for d in data])
    dones = np.array([d[4] for d in data])

    plt.hist(rewards, bins=50)
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.title('Reward distribution')
    plt.show()

if __name__ == '__main__':
    config = dv2.defaults.update({
    'logdir': '~/logdir/minigrid',
    'log_every': 1e3,
    'train_every': 10,
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
}).parse_flags()
    env = gym.make('BipedalWalker-v3')
    dv2.train(env, config)
    #a = data_collection(100)
    #print(a)
    #main()
    """
    # Train the model
    model = BipedalWalkerModel().to(device)
    train(model, num_epochs=200, batch_size=32, learning_rate=0.001, device=device)

    # Test the model
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs_tensor = torch.from_numpy(np.array(obs)).float().to(device)
        action_tensor = model(obs_tensor)
        action = action_tensor.detach().cpu().numpy()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

    print("Total reward: %f" % total_reward)
    """