import numpy as np
import math 

from . import Drone
from . import GCRFModel as M
from . import constants as C

class Swarm():
	def __init__(self, num_drones, rnd_state, num_training_steps, num_inference_steps, use_structure, shape="cube"):
		self.N = num_drones
		self.drones = []
		self.training  = True
		self.use_model = True
		self.using_expansion = False # For experiments
		self.pred_horz = 0
		self.expansion_timer = 0
		self.expansion_state = C.EXP_OFF
		self.color = "k" # Black
		
		# Global timestep counter
		self.timestep = 0
		self.num_training_steps = num_training_steps
		self.T_train = num_training_steps - C.WINDOW_SIZE

		self.wind_training_timestep  = rnd_state.randint(0, num_training_steps-C.LENGTH_MEAN)
		self.wind_inference_timestep = rnd_state.randint(num_training_steps+5, num_training_steps+num_inference_steps - C.LENGTH_MEAN)
		self.wind_dev = np.asarray([0,0,0])
		self.wind_scales = [] # One for each drone, gets filled in by the wind sampling.

		# Current Dataset
		self.data_window = []
		self.curr_X = None
		self.curr_Y = None
		
		# Historical Data	
		self.X = np.zeros((self.N, 3*C.WINDOW_SIZE, self.T_train), dtype=float)
		self.Y = np.zeros((self.N, 3, self.T_train), dtype=float)
		
		# GCRF Model.
		self.model = M.GCRFModel()
		self.S     = None          # Similarity Matrix for GCRF Model
		self.use_structure = use_structure
		print('self.use_structure:', self.use_structure)
		
		if shape == "cube":
			# For now, if cube, assume num_drones has a perfect
			# cube root. 
			side_len = int(num_drones ** (1/3))
			self.s = side_len
			# Create adjacency and similarity matrices.
			self.G = np.ones( (side_len**3, side_len**3), dtype=int) # Change this to change network
			self.S = np.zeros((side_len**3, side_len**3), dtype=float)

			for layer_number in range(side_len):
				z_loc = C.SEPARATION * layer_number
				for row in range(side_len):
					x_loc = C.SEPARATION * row
					for col in range(side_len):
						y_loc = C.SEPARATION * col
						d = Drone.Drone()
						d.pos = np.asarray([x_loc, y_loc, z_loc])
						d.target = d.pos
						d.init_PIDs()
						self.drones.append(d)
			
		if shape == "planar":
			side_len = int(num_drones**(1/2))
			self.s = side_len
			self.G = np.ones( (side_len**2, side_len**2), dtype=int) # Change this to change network
			self.S = np.zeros((side_len**2, side_len**2), dtype=float)

			z_loc = 0
			for row in range(side_len):
				x_loc = C.SEPARATION * row
				for col in range(side_len):
					y_loc = C.SEPARATION * col
					d = Drone.Drone()
					d.pos = np.asarray([x_loc, y_loc, z_loc])
					d.target = d.pos
					d.init_PIDs()
					self.drones.append(d)

		self.update_S()

	#### "Public" Methods #########
	def tick(self, wind):
		# MODIFIED DRONE SPECIFIC SAMPLING
		self.wind_dev = [dev * C.DT for dev in wind.sample_wind(self.drones)]

		'''
		# 'State Machine' for expansion procedure
		if self.using_expansion and not self.training:	
			if self.expansion_state == C.EXP_OFF:
				self.expansion_timer += 1
				if self.expansion_timer > self.pred_horz:
					self.exp_hover()
					self.expansion_state = C.EXP_HOVER
					print("Expansion: drones switch to hover mode")
			elif self.expansion_state == C.EXP_HOVER:
				# Check if drones are at targets
				if self.drones_at_targets():
					self.exp_expand()
					self.expansion_state = C.EXP_EXPANDING
					print("Expansion: drones expanding")
			elif self.expansion_state == C.EXP_EXPANDING:
				if self.drones_at_targets():
					self.exp_correct_targets()
					self.expansion_state == C.EXP_OFF
					self.expansion_timer = 0
					print("Expansion: drones update targets")
		'''
		# if self.timestep == self.wind_training_timestep:
		# 	self.wind_dev = wind.sample_wind() * C.DT
		# elif self.timestep == self.wind_inference_timestep:
		# 	self.wind_dev = wind.sample_wind() * C.DT
		# elif (self.timestep > self.wind_training_timestep + wind.gust_length and self.timestep < self.wind_inference_timestep) or \
		# 	   (self.timestep > self.wind_inference_timestep + wind.gust_length):
		# 		self.wind_dev = np.asarray([0, 0, 0])
		# if ( self.timestep >= self.wind_training_timestep and self.timestep <= self.wind_training_timestep + wind.gust_length ) or \
		#    ( self.timestep >=  self.wind_inference_timestep and self.timestep <= self.wind_inference_timestep + wind.gust_length ):
		# 	#print (self.timestep, '\t', self.wind_dev)

		for i in range(len(self.drones)):
			# print(self.drones[i].pos, self.wind_dev[i])
			self.drones[i].pos += self.wind_dev[i]

		if self.training:
			for d in self.drones:
				d.update_state_from_pos(d.pos)
				d.pos_estimate = np.copy(d.pos)
				
				d.pos += d.vel*C.DT
				d.H_pos.append(np.copy(d.pos))
				
				d.update_state_from_pos_estimate(d.pos_estimate)
				d.pos_estimate += d.vel_estimate*C.DT
				#d.H_pos_estimate.append(np.copy(d.pos_estimate))
		
		
		self.update_data()

		# INFERENCE
		if not self.training:
			
			for index, d in enumerate(self.drones):
				
				
				#d.update_inference(self.model, self.use_model, self.curr_X, self.S, index, self.use_structure)
				
				if self.use_model:
					# Apply output of model to predict deviation from Wind
					# self.pos_estimate += model.predict(X, S, d_index) # I think? more on this later.
					d.pos_estimate = self.model.predict(self.curr_X, self.S, index, self.use_structure) # I think? more on this later.
				
				# In inference, we use the ESTIMATE of pos to update
				d.update_state_from_pos_estimate(d.pos_estimate)

				# Finally, update current estimate of position
				d.pos_estimate += d.vel_estimate*C.DT
				d.H_pos_estimate.append(np.copy(d.pos_estimate))
				
				# ============================================================================
				
				# We also update the 'real' position, because the real position
				# has already been moved by the wind, the effect of the PID's 
				# changes to the acceleration are still impacting the true
				# location ONTOP of the wind moving it. 
				d.update_state_from_pos(d.pos)
				
				d.pos += d.vel*C.DT
				d.H_pos.append(np.copy(d.pos))
				
				

		#print('self.timestep:', self.timestep)

		# Data Gathering/Model Training
		if self.training:
			#self.update_data(wind_dev)

			if self.timestep == self.num_training_steps - 1:
				self.model.train(self.S, self.X, self.Y)
				# print('theta:', self.model.theta)
				# pass
		
		self.timestep += 1
				
	def set_swarm_target_relative(self, dpos):
		delta = np.asarray(dpos)
		for d in self.drones:
			d.target = d.pos + delta
			# d.init_PIDs() 
	
	def set_swarm_pos_relative(self, dpos):
		delta = np.asarray(dpos)
		for d in self.drones:
			d.pos = d.pos + delta

	# Should be called when we change the target.
	def init_drone_PIDs(self):
		for d in self.drones:
			d.init_PIDs()

	######################

	#### Expansion Methods ########
	def use_expansion(self, pred_horz):
		self.using_expansion = True
		self.pred_horz = pred_horz
		self.expansion_timer = pred_horz # So we run immediatly the first time

	def exp_hover(self):
		# Save current target and set target to current pos estimate. 
		for d in self.drones:
			d.saved_target = d.target
			d.set_target(d.pos_estimate)

	def exp_expand(self):
		# Calculate 'center' of the swarm
		poss = []
		for d in self.drones:
			poss.append(d.pos_estimate)
		center = np.mean(poss, axis=0)

		# TODO - Calculate/determine 'max' variance, use
		# to determine the magnitude of expansion vector

		# Move each drone away from center
		for d in self.drones:
			delta = d.pos_estimate - center
			d.exp_vector = (delta / np.linalg.norm(delta))
			d.exp_vector *= C.TEST_VAR_RADIUS
			d.set_target(d.pos_estimate+d.exp_vector)

	def exp_correct_targets(self):
		for d in self.drones:
			d.set_target(d.saved_target+d.exp_vector)

	def drones_at_targets(self):
		have_reached = True
		for d in self.drones:
			have_reached = have_reached and d.has_reached_target(C.TARGET_EPSILON)
		return have_reached
	#######################

	#### Model Methods ###########
	def update_S(self):
		I, J = np.shape(self.S)
		for i in range(I):
			for j in range(J):
				if j > i and self.G[i][j] == 1:
					# Similarity is their distance
					d_i = self.drones[i]
					d_j = self.drones[j]
					self.S[i][j] = np.linalg.norm(d_i.pos - d_j.pos)
					self.S[j][i] = self.S[i][j]
					
		# Threshold S
		self.S = self.S / np.max(self.S)
		self.S = 1.0 - self.S
		for i in range(I):
			for j in range(J):
				if self.S[i][j] <= C.SEPARATION:
					self.S[i][j] = 0.0

	def update_G(self):
		# In the future, we will have some threshold distance that
		# defines when drones are neighbors, and can therefore 'share'
		# info, or be considered connected. 
		pass

	def update_data(self):		
		curr_positions = []
		if (self.timestep - C.WINDOW_SIZE) < self.T_train:
			for d in self.drones:
				curr_positions.append(d.pos)
		else:
			for d in self.drones:
				curr_positions.append(d.pos_estimate)		
				
		self.data_window.append(np.copy(curr_positions))
		
		if len(self.data_window) > C.WINDOW_SIZE:
			
			self.curr_X = np.zeros((self.N, 3*C.WINDOW_SIZE), dtype=float)
			for k in range(0,C.WINDOW_SIZE):
				for i in range(0,self.N):
					self.curr_X[i, k*3] = self.data_window[k][i][0]
					self.curr_X[i, k*3+1] = self.data_window[k][i][1]
					self.curr_X[i, k*3+2] = self.data_window[k][i][2]
					
			self.curr_Y = np.zeros((self.N, 3), dtype=float)
			for i in range(0,self.N):
				self.curr_Y[i, 0] = self.data_window[-1][i][0]
				self.curr_Y[i, 1] = self.data_window[-1][i][1]
				self.curr_Y[i, 2] = self.data_window[-1][i][2]
			
			if (self.timestep - C.WINDOW_SIZE) < self.T_train:
				self.X[:, :, self.timestep-C.WINDOW_SIZE] = np.copy(self.curr_X)
				self.Y[:, :, self.timestep-C.WINDOW_SIZE] = np.copy(self.curr_Y)
			
			self.data_window.pop(0)
		
	######################
	def dump_state(self):
		print(np.shape(self.data_x), np.shape(self.hdata_x))

	def dump_locations(self, return_true_positions):
		if return_true_positions:
			return np.asarray([d.pos for d in self.drones])
		else:
			return np.asarray([d.pos_estimate for d in self.drones])


# Going to need these later. For now, assuming fully connected G so
# don't need them really. 
def index_from_spatial(x, y, z, s):
	return x + y*s + z*(s*s)

def spatial_from_index(n, s):
	z = math.floor(n / (s*s))
	y = math.floor((n/s) - z*(s*s))
	x = n - y*s - z*s*s
	return [x,y,z]
