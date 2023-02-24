
class ActionPerformer:
    def __init__(self, action):
        self.action = action

    def perform(self):
        self.action.perform()