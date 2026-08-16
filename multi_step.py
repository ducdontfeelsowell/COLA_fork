from collections import deque
import numpy as np
from base import MOMemory


class MultiStepBuff:
    keys = ["state", "preference", "action", "reward"]

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.memory = {
            key: deque(maxlen=self.maxlen)
            for key in self.keys
            }

    def append(self, state,preference, action, reward):
        self.memory["state"].append(state)
        self.memory["preference"].append(preference)
        self.memory["action"].append(action)
        self.memory["reward"].append(np.asarray(reward, dtype=np.float32))

    def get(self, gamma=0.99):
        if len(self) == 0:
            raise IndexError("cannot read from an empty MultiStepBuff")
        reward = self._multi_step_reward(gamma)
        preference = self.memory["preference"].popleft()
        state = self.memory["state"].popleft()
        action = self.memory["action"].popleft()
        _ = self.memory["reward"].popleft()
        return state, preference, action, reward

    def _multi_step_reward(self, gamma):
        return np.sum([
            r * (gamma ** i) for i, r
            in enumerate(self.memory["reward"])], axis=0)

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f'There is no key {key} in MultiStepBuff.')
        return self.memory[key]

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def __len__(self):
        return len(self.memory['state'])


class MOMultiStepMemory(MOMemory):

    def __init__(self, capacity, state_shape, reward_shape, action_shape, device,
                 gamma=0.99, multi_step=3):
        super(MOMultiStepMemory, self).__init__(
            capacity, state_shape, reward_shape, action_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, preference, action, reward, next_state, done,
               episode_done=False):
        if self.multi_step != 1:
            self.buff.append(state, preference, action, reward)

            if len(self.buff) == self.multi_step:
                state, preference, action, reward = self.buff.get(self.gamma)
                self._append(state, preference, action, reward, next_state, done)

            if episode_done or done:
                # Preserve the shorter n-step returns at the end of an episode.
                # The full-length item above has already popped the first entry.
                while len(self.buff) > 0:
                    state, preference, action, reward = self.buff.get(self.gamma)
                    self._append(
                        state, preference, action, reward, next_state, done)
                self.buff.reset()
        else:
            self._append(state, preference, action, reward, next_state, done)
