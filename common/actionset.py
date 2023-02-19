import time
import random

TICK_TIME = 0.6

class Agent:
    def __init__(self, action_set):
        self.action_set = action_set

    def act(self):
        # select up to 4 actions for the current tick
        actions = self.action_set.select_actions()
        
        # perform the selected actions
        for action in actions:
            action.perform()

num_monsters = 0

class ActionSet:
    def __init__(self):
        # Define the possible actions
        self.actions = []

        # Movement actions
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    self.actions.append(('move', (x, y, z)))

        # Attack actions
        for i in range(num_monsters):
            self.actions.append(('attack', i))

        # Inventory actions
        for i in range(28):
            self.actions.append(('use_item', i))
            self.actions.append(('drop_item', i))
            self.actions.append(('equip_item', i))
            self.actions.append(('unequip_item', i))

        # Ability actions
        self.actions.append(('toggle_ability', 'ability_1'))
        self.actions.append(('toggle_ability', 'ability_2'))
        self.actions.append(('toggle_ability', 'ability_3'))
        self.actions.append(('toggle_ability', 'ability_4'))

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.actions[idx]

    def select_actions(self, state):
        # Sample up to 4 actions
        actions = random.sample(self.actions, min(4, len(self.actions)))

        # Modify the move action to take an argument for which tile to move to
        for i, action in enumerate(actions):
            if action[0] == 'move':
                # Select a random tile to move to
                move_to_tile = random.choice(state['tiles'])
                actions[i] = ('move', move_to_tile)

            # Modify the use item action to choose one of 28 items from the inventory
            elif action[0] == 'use_item':
                # Choose a random item from the inventory to use
                item_idx = random.randrange(28)
                actions[i] = ('use_item', item_idx)

            # Modify the attack action to take an argument for which monster to attack
            elif action[0] == 'attack':
                # Select a random monster to attack
                monster_idx = random.randrange(num_monsters)
                actions[i] = ('attack', monster_idx)

        return actions


class Action:
    def __init__(self):
        self.available = True

    def is_available(self):
        # check if the action is available to be performed
        return self.available

    def perform(self):
        # perform the action
        pass
