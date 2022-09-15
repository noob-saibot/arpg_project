import gym
from gym import spaces
import pygame
import numpy as np
from enum import Enum
from functools import reduce


class ActionSpace(Enum):
    TurnR = 0
    TurnL = 1
    MoveF = 2
    MoveB = 3
    Touch = 4
    Jump = 5


class Actions(Enum):
    TurnR = 90
    TurnL = -90
    MoveF = 1
    MoveB = -1
    Touch = None
    Jump = None


class Box(spaces.Box):
    @property
    def size(self):
        return reduce(lambda x, y: x*y, self.shape)


class Discrete(spaces.Discrete):
    @property
    def size(self):
        return 1


class SimpleArenaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": Box(0, size - 1, shape=(2,), dtype=int),
                "target": Box(0, size - 1, shape=(2,), dtype=int),
                "agent_direction": Discrete(4, start=0),
                "velocity": Discrete(4, start=-1)
            }
        )

        # We have 4 actions, corresponding to "turn right", "turn left", "move forward", "touch", "jump"
        self.action_space = spaces.Discrete(5)
        self.step_counter = 0

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        # self._action_to_direction = Actions

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.distance_prev = None
        self.distance_curr = None
        self._agent_location = None
        self._target_location = None

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    @property
    def shape(self):
        return np.sum([space.size for space in self.observation_space.values()])

    def _get_obs(self):
        return {
            "agent": self._agent_location.astype(np.int64),
            "agent_direction": self._agent_direction,
            "target": self._target_location.astype(np.int64),
            "velocity": self._move_speed,
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(
            0, self.size, size=2, dtype=int
        ).astype(np.int32)
        self._agent_direction = self.np_random.choice(list([0, 90, 180, 270])).astype(
            np.int64
        )
        self._move_speed = np.int64(0)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.step_counter += 1
        action = ActionSpace(action)
        direction = Actions[ActionSpace(action).name].value
        self.distance_prev = self._get_distance(self._agent_location, self._target_location)

        if action in [ActionSpace.TurnL, ActionSpace.TurnR]:
            self._agent_direction = (self._agent_direction - direction) % 360
            self._move_speed = 0
        elif action in [ActionSpace.MoveF, ActionSpace.MoveB]:
            # We use `np.clip` to make sure we don't leave the grid
            self._move_speed = np.clip(self._move_speed + direction, -1, 2)
            self._agent_location = np.clip(
                self._agent_location
                - np.array(
                    [
                        -1
                        * int(np.cos(self._agent_direction * np.pi / 180))
                        * self._move_speed,
                        int(np.sin(self._agent_direction * np.pi / 180))
                        * self._move_speed,
                    ]
                ),
                0,
                self.size - 1,
            )
        self.distance_curr = self._get_distance(self._agent_location, self._target_location)

        # An episode is done iff the agent has reached the target
        terminated = (
            np.array_equal(self._agent_location, self._target_location)
            and action == ActionSpace.Touch
        )
        reward = self._calculate_reward(action) - self.step_counter % 2  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _calculate_reward(self, action):
        same_lock = np.array_equal(self._agent_location, self._target_location)
        if same_lock and action == ActionSpace.Touch:
            return 10
        if same_lock:
            return 5
        if action in [ActionSpace.TurnL, ActionSpace.TurnR]:
            if same_lock:
                return -2
            if self._same_quadrant():
                return 1
            return -1
        if action in [ActionSpace.MoveF, ActionSpace.MoveB]:
            if self.distance_curr < self.distance_prev:
                if self._move_speed == 2:
                    return 2
                return 1
            else:
                if self._move_speed == 2:
                    return -2
                return -1
        if action == ActionSpace.Touch:
            return -1

    @staticmethod
    def _get_distance(point1, point2):
        temp = point1 - point2
        sum_sq = np.dot(temp.T, temp)
        return np.sqrt(sum_sq)

    def _same_quadrant(self):
        if self._agent_direction == 0:
            view_quadrants = {1, 2}
        elif self._agent_direction == 90:
            view_quadrants = {4, 1}
        elif self._agent_direction == 180:
            view_quadrants = {3, 4}
        else:
            view_quadrants = {2, 3}

        diff = self._agent_location - self._target_location
        if diff[0] < 0 and diff[1] > 0:
            target_quadrant = 1
        elif diff[0] < 0 and diff[1] < 0:
            target_quadrant = 2
        elif diff[0] > 0 and diff[1] < 0:
            target_quadrant = 3
        else:
            target_quadrant = 4
        return target_quadrant in view_quadrants

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.arc(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                pix_square_size * self._agent_location,
                (pix_square_size, pix_square_size),
            ),
            self._agent_direction * np.pi / 180 - 0.7,
            self._agent_direction * np.pi / 180 + 0.7,
            3
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
