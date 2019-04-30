from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from .model import BoidFlockers
from .SimpleContinuousModule import SimpleCanvas

def boid_draw(agent):
    return {"Shape": "circle", "r": 4, "Filled": "true", "Color": "Red"}

boid_canvas = SimpleCanvas(boid_draw, 500, 500)
model_params = {
    "population": 200,
    "width": 100,
    "height": 100,
    "speed": 5,
    "vision": 10,
    "separation": 2,
    "cohere": UserSettableParameter('slider', 'Coherence', 0.2,0.0,1,0.1,
                                    description='coherence'),
}

server = ModularServer(BoidFlockers, [boid_canvas], "Boids", model_params)
