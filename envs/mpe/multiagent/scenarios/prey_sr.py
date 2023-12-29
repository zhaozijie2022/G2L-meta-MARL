import numpy as np
from envs.mpe.multiagent.core import World, Agent, Landmark
from envs.mpe.multiagent.scenario import BaseScenario

def bound(x):
    x = abs(x)
    if x < 0.9:
        return 0
    if x < 1.0:
        return (x - 0.9) * 10
    return min(np.exp(2 * x - 2), 10)


class Scenario:
    def make_world(self, n_predators, n_preys, n_obstacles):
        world = World()
        world.bb = 1.2
        world.boundary = [np.array([world.bb, 0]), np.array([-world.bb, 0]),
                          np.array([0, world.bb]), np.array([0, -world.bb])]
        world.wall = [np.array([world.bb, world.bb]), np.array([-world.bb, world.bb]),
                        np.array([-world.bb, -world.bb]), np.array([world.bb, -world.bb])]
        # set any world properties first
        world.dim_c = 2
        world.target_id = 0  # 追逐哪个prey
        # add agents
        predators = [Agent() for _ in range(n_predators)]
        world.n_predators = n_predators
        for i, predator in enumerate(predators):
            predator.name = 'predator %d' % i
            predator.size = 0.1
            predator.adversary = True
            predator.target = False
            predator.accel = 3.0
            predator.max_speed = 1.0

        preys = [Agent() for _ in range(n_preys)]
        world.n_preys = n_preys
        for i, prey in enumerate(preys):
            prey.name = 'prey %d' % i
            # prey.size = 0.04 + 0.01 * i
            prey.size = 0.1
            prey.adversary = False
            prey.target = True if i == world.target_id else False
            prey.accel = 4.0
            prey.max_speed = 1.3

        world.agents = predators + preys
        for i, agent in enumerate(world.agents):
            agent.collide = True
            agent.silent = True

        world.landmarks = [Landmark() for _ in range(n_obstacles)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            if agent.target:
                agent.color = np.array([0.85, 0.35, 0.35])
                agent.collide = True
            elif agent.adversary:
                agent.color = np.array([0.35, 0.85, 0.35])
                agent.collide = False
            else:
                agent.color = np.array([0.35, 0.35, 0.85])
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        if dist < dist_min:
            return True
        else:
            return False

    def preys(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    def predators(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # return 0.
        vertices = [pred.state.p_pos for pred in self.predators(world)]
        target = [prey for prey in self.preys(world) if prey.target][0]
        point = target.state.p_pos
        reward = 0.
        # reward += is_inside_triangle(vertices, point) / circumference(vertices)
        reward += - 0.1 * np.linalg.norm(point - agent.state.p_pos)
        # reward += np.sum([bound(agent.state.p_pos[p]) for p in range(world.dim_p)])
        reward += 10 * self.is_collision(agent, target)
        return reward

    def observation(self, agent, world):
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        other_pos = []
        other_vel = []
        other_dist = []
        for other in world.agents:
            if other is agent:
                continue

            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_dist.append([np.linalg.norm(other.state.p_pos - agent.state.p_pos)])

        return np.concatenate([agent.state.p_pos] + entity_pos + other_pos + other_dist + other_vel)
