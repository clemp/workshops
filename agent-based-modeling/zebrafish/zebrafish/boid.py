import numpy as np
import mesa
from mesa import Agent

class Boid(Agent):
    '''
    A Boid-style zebrafish agent.

    The agent follows ... behaviors to find its prey.
    '''

    def __init__(self, unique_id, model, pos, speed, velocity, vision,
        separation, cohere=0.25, separate=0.25, match=0.04):
        '''
        Create a new Boid zebrafish agent.

        Args:
            unique_id: unique agent identifier
            pos: starting position
            speed: distance to move per step
            heading: numpy vector for the Boid's direction of movement.
            vision: radius to look around for nearby Boids.
            separation: minimum distance to maintain from other Boids.
            cohere: relative importance of matching neighbors' positions
            separate: the relative importance of avoiding close neighbors
            match: the relative importance of matching neighbors' headings
        '''
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.speed = speed
        self.velocity = velocity
        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match

    def cohere(self, neighbors):
        '''
        Return the vector toward the center of the mass of the local neighbors.
        '''
        cohere = np.zeros(2)
        if neighbors:
            for neighbor in neighbors:
                cohere += self.model.space.get_heading(self.pos, neighbor.pos)
            cohere /= len(neighbors)
        return cohere

    def separate(self, neighbors):
        '''
        Return a vector away from any neighbors closer than separation dist
        '''
        me = self.pos
        them = (n.pos for n in neighbors)
        separation_vector = np.zeros(2)
        for other in them:
            if self.model.space.get_distance(me, other) < self.separation:
                separation_vector -= self.model.space.get_heading(me, other)
        return separation_vector

    def match_heading(self, neighbors):
        '''
        Return a vector of the neighbors' average heading.
        '''
        match_vector = np.zeros(2)
        if neighbors:
            for neighbor in neighbors:
                match_vector += neighbor.velocity
            match_vector /= len(neighbors)
        return match_vector

    def step(self):
        '''
        Get the boid's neighbors, compute the new vector, and move accordingly.
        '''
        neighbors = self.model.space.get_neighbors(self.pos, self.vision, False)
        self.velocity += (
            self.cohere(neighbors) * self.cohere_factor +
            self.separate(neighbors) * self.separate_factor +
            self.match_heading(neighbors) * self.match_factor
            ) / 2
        self.velocity /= np.linalg.norm(self.velocity)
        new_pos = self.pos + self.velocity * self.speed
        self.model.space.move_agent(self, new_pos)
