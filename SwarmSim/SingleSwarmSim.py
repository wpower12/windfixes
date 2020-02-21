import numpy as np

from . import Swarm as S
from . import Wind  as W
from . import SingleSwarmAnimator as A

class SingleSwarmSim():
	def __init__(self, swarm, rnd_seed, num_training_steps, num_inference_steps, use_structure, fig_title, wind_windows, animate=True):

		num_drones, swarm_type, color, pos, target = swarm
		self.prng = np.random.RandomState(rnd_seed)
		self.sim = S.Swarm(num_drones, self.prng, num_training_steps, num_inference_steps, use_structure, swarm_type)
		self.sim.color = color
		self.sim.set_swarm_pos_relative(pos)
		self.sim.set_swarm_target_relative(target)
		self.sim.init_drone_PIDs()

		# Windows! Set the

		self.wind = W.Wind(self.prng, wind_windows)

		self.in_training = True
		self.plot_errors = False
		self.animating = animate
		if self.animating:
			self.anm = A.SingleSwarmAnimator(fig_title)
		
	def animate(self):
		self.anm.plot_drones(self.sim, 
							self.sim.training, 
							self.plot_errors)

	def tick(self):
		if self.animating:
			self.animate()
		self.sim.tick(self.wind)

	def start_inference(self, use_model=True):
		self.sim.training = False
		self.sim.use_model = use_model

	def use_expansion(self, ph):
		self.sim.use_expansion(ph)

	def dump_drone_locations(self, return_true_positions):
		return self.sim.dump_locations(return_true_positions)
