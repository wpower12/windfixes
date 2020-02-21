import numpy as np
from . import constants as C

class Wind():
	def __init__(self, rnd_state, windows):
		self.gusting = False
		self.gust_length = 0 # How many ticks (state-advances) a gust lasts
		self.gust_timer  = 0 # How long a gust has been going
		self.timer = 0 # General timer, same as sim timer

		# Storing these for debugging, but not strictly needed?
		self.gust_angle  = 0 # Which Direction
		self.gust_mag    = 0 # How hard

		# The above 2 will resolve into a single 3 vector
		self.gust_vect = None

		# Lets us ensure we have the same state between invocations
		self.prng = rnd_state

		# The windows tell us when to start and stop gusting.
		self.windows = windows
		self.cur_window = 0

	def sample_wind(self, drones):
		self.advance_state()
		if self.gusting:
			return self.scaled_wind_vectors(drones)
		else: # Not gusting
			return np.zeros((len(drones), 3))

	def advance_state(self):
		if self.gusting:
			if self.timer > self.windows[self.cur_window][1]:
				self.gusting = False
				self.gust_vect= np.asarray([0, 0, 0])
				self.cur_window += 1

		else: # Not Gusting
			if self.cur_window < len(self.windows) and self.timer > self.windows[self.cur_window][0]:
				self.gusting = True
				self.gust_length = self.sample_length()
				self.gust_angle  = self.sample_angle()
				self.gust_mag    = self.sample_mag()
				self.gust_vect   = self.resolve_vector()

		self.timer += 1

	def scaled_wind_vectors(self, drones):
		positions = np.asarray([d.pos for d in drones])
		projections = np.dot(positions, -1 * self.gust_vect)

		scale_range = C.MAX_SCALING - C.MIN_SCALING

		max_proj = projections.max()
		min_proj = projections.min()
		proj_range = max_proj - min_proj

		return [((p - min_proj) * scale_range / proj_range) + C.MIN_SCALING for p in projections]

	# Keeping the sampling split out in case I want to do anything
	# 'extra' with them before/after sampling.
	def sample_start(self):
		return self.prng.binomial(1, C.START_P)

	def sample_length(self):
		# TODO - Check if this is an ok way to get a 'discrete' normal
		return int(self.prng.normal(C.LENGTH_MEAN, C.LENGTH_VAR))

	def sample_angle(self):
		return self.prng.normal(C.ANGLE_MEAN, C.ANGLE_VAR)

	def sample_mag(self):
		return self.prng.normal(C.MAG_MEAN, C.MAG_VAR)

	def resolve_vector(self):
		x = self.gust_mag * np.cos(self.gust_angle)
		y = self.gust_mag * np.sin(self.gust_angle)
		return np.asarray([x, y, 0.0])
