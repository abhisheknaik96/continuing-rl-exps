# """
#         This file presents Adaptive Optics Gym environment.
#         Author: Payam Parvizi
# """

# # from hcipy import *
# from hcipy import imshow_field, make_pupil_grid, \
#     make_circular_aperture, make_focal_grid, FraunhoferPropagator, \
#     Wavefront, make_zernike_basis, ModeBasis, DeformableMirror, \
#     make_disk_harmonic_basis, Cn_squared_from_fried_parameter, \
#     make_pupil_grid, StepIndexFiber, get_strehl_from_focal, InfiniteAtmosphericLayer, \
#     large_poisson
    
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import scipy.ndimage as ndimage
# import matplotlib.pyplot as plt

# from skimage.metrics import structural_similarity as ssim
# import warnings
# warnings.simplefilter("ignore")


# class AOEnv(gym.Env):
#     def __init__(self, 
#                  atm_type='quasi_static',               # atmospheric condition: 'quasi_static, semi_dynamic, dynamic
#                  atm_vel = 0,                           # atmosphere velocity
#                  atm_fried = 0.10,                      # Fried parameter of the atmosphere
#                  act_type = 'zernike',            # action type: 'num_actuators', 'zernike'
#                  act_dim = 9,                          # action dimension
#                  obs_dim = 5,                           # observation dimension   
#                  rew_type = 'smf_ssim',             # reward type: 'strehl_ratio', 'smf_ssim'
#                  rew_threshold = None,                  # Threshould of the reward value
#                  timesteps_per_episode= 20,             # Number of timesteps per episode
#                  flat_mirror_start_per_episode = 1,  # If we want each episode to start with flat mirror
#                  SH_operation = 0,                  # If we require Shack_Hartmann wavefront sensor operation
#                  delta_t = 1e-4,
#                  c_act_range = 10,
#                  c_rand = 1,
#                  c_mult = 0.2,
#                  c_rew = 0.8,
#                  c_mode1 = 1,
#                  ar_case = 1,
#                  seed_v = 12,
#                  layer_no = 0,
#                  ): 
        
#         super(AOEnv, self).__init__()
        
#         self.atm_type = atm_type
#         self.rew_type = rew_type
#         self.act_type = act_type
#         self.flat_mirror_start_per_episode = flat_mirror_start_per_episode

#         self.rew_threshold = rew_threshold
#         self.SH_operation = SH_operation
#         self.delta_t = delta_t
#         self.obs_max = 3.017
#         self._c_act_range = c_act_range
#         self._c_rand = c_rand
#         self._c_mult = c_mult
#         self._c_rew = c_rew
#         self._c_mode1 = c_mode1
#         self._ar_case = ar_case
#         self._seed = seed_v
#         self._layer_no = layer_no
        
#         # Parameters used in the environment:
#         self.parameters_init(act_dim, atm_vel, obs_dim, timesteps_per_episode, atm_fried)
    
#         # Extract out dimensions of observation and action spaces:
#         self.observation_space = spaces.Box(low=0, high=self.obs_max, shape=(self.num_focal_pixels_fiber_subsample**2, ), dtype=np.float16)
#         self.action_space = spaces.Box(low=-self._c_act_range, high=self._c_act_range, shape=(act_dim, ), dtype=np.float16)
        
#         # simulating the pupil:
#         # Modeling the the diameter of the pupil of a telescope as a function of the telescope's diameter
#         aperture, pupil_grid  = self.pupil_simulation()

#         # Incoming wavefront:
#         # simulation of the incoming wavefront from the satellite
#         focal_grid = self.incoming_wavefront(aperture, pupil_grid)
        
#         # Deformable mirror function:
#         dm_modes = self.DM_function(act_type, pupil_grid)
        
#         # Atmospheric Turbulence:
#         # Simulating the atmosphere
#         self.atmospheric_turbulence(pupil_grid)

#         # Fiber coupling:
#         self.fiber_coupling()
        
#         # Shack-Hartmann wavefront sensor initialization
#         if self.SH_operation == 2:
#             self.shack_hartmann_init(pupil_grid, focal_grid, aperture, dm_modes)

#         self.timestep = 0
#         self.episode_no = 0    
        
        
#     def reset(self, seed=None, options=None):

#         if self.atm_type == 'semi_dynamic':
#             self.layer.reset()
        
#         if self.flat_mirror_start_per_episode == 1:
#             self.deformable_mirror.flatten()
        
#         # start the environment at time 0 sec
#         self.timestep_render = 0
#         self.layer.t = self.timestep * self.delta_t

#         # get the phase screen for plot
#         phase_screen_phase = self.layer.phase_for(self.wavelength_wfs)    # Get the phase screen in radians
#         self.phase_screen_opd = phase_screen_phase * (self.wavelength_wfs / (2 * np.pi)) * 1e6

#         # Propagatation of wavefront through atmosphere
#         wf_wfs_after_atmos = self.layer(self.wf_wfs_fiber)

#         # Propagatation of wavefront through deformable mirror
#         wf_wfs_after_dm = self.deformable_mirror(wf_wfs_after_atmos)

#         # Propagatation of wavefront through focal plane before fiber
#         self.wf_wfs_after_foc = self.propagator_fiber(wf_wfs_after_dm)
#         self.wf_wfs_after_foc_subsample = self.propagator_fiber_subsample(wf_wfs_after_dm)     # subsampled for photodetector

#         # The observation - Power of the wavefront propagated through the focal plane
#         state = self.wf_wfs_after_foc_subsample.power

#         return np.array(state, dtype=np.float16), {}


#     def step(self, action):
        
#         done = False
#         trunc = False 
        
#         """
#         Shack-Hartmann creates normalized action which can be used directly in this function,
#         However, the action generated by Actor needs to be normalized in here
#         """
        
#         if self.SH_operation == 2:
#             self.deformable_mirror.actuators = self.SH_step()

#         elif self.SH_operation == 0:

#             arange = np.array([0,0,1,1,1,2,2,2,2])
                
#             action = action / (arange + int(self._c_rand))
#             self.deformable_mirror.actuators = np.append(np.array([int(self._c_mode1)]), action)
#             #self.deformable_mirror.actuators = action
#             self.deformable_mirror.actuators *= self._c_mult * self.wavelength_sci / (np.std(self.deformable_mirror.surface))

#         elif self.SH_operation == 1:
#             self.deformable_mirror.flatten()
        
#         # self.deformable_mirror.flatten()
#         # The next time of the atmospheric layer
#         self.timestep += 1
#         self.timestep_render += 1
#         self.layer.t = self.timestep * self.delta_t
        
#         # get the phase screen for plot
#         phase_screen_phase = self.layer.phase_for(self.wavelength_wfs)    # Get the phase screen in radians
#         self.phase_screen_opd = phase_screen_phase * (self.wavelength_wfs / (2 * np.pi)) * 1e6

#         # Propagatation of wavefront through atmosphere
#         wf_wfs_after_atmos = self.layer(self.wf_wfs_fiber)

#         # Propagatation of wavefront through deformable mirror
#         wf_wfs_after_dm = self.deformable_mirror(wf_wfs_after_atmos)
        
#         # Propagatation of wavefront through focal plane before fiber
#         self.wf_wfs_after_foc = self.propagator_fiber(wf_wfs_after_dm)
#         self.wf_wfs_after_foc_subsample = self.propagator_fiber_subsample(wf_wfs_after_dm)     # subsampled for photodetector

#         # The observation - Power of the wavefront propagated through the focal plane
#         next_state = self.wf_wfs_after_foc_subsample.power

#         reward, rew_fiber = self.reward_function(next_state)
        
#         # check if done or not:
        
#         if self.timestep_render == self.max_steps:
#             trunc = True
#             # print(rew_fiber)
#             self.episode_no += 1
#         else:
#             trunc = False

#         return np.array(next_state, dtype=np.float16), float(reward), done, trunc, {"power":float(rew_fiber)}


#     def render(self, close=False):
        
#         plt.suptitle('episode %d - timestep %d / %d' % (self.episode_no+1, self.timestep_render+1, self.max_steps))

#         # plot of the atmosphrere
#         plt.subplot(2,2,1)
#         plt.title('Atmosphere')
#         imshow_field(self.phase_screen_opd, vmin=-6, vmax=6, cmap='RdBu')
#         plt.colorbar()

#         # plot of the wavefront power after focal plane before fiber
#         plt.subplot(2,2,3)
#         imshow_field(self.wf_wfs_after_foc.power)
#         circ = plt.Circle((0, 0), self.singlemode_fiber_core_radius, edgecolor='white', fill=False, linewidth=2, alpha=0.5)
#         plt.gca().add_artist(circ)
#         plt.xlabel('x (um)')
#         plt.ylabel('y (um)')
#         plt.colorbar()
        
#         plt.subplot(2,2,4)
#         imshow_field(self.wf_wfs_after_foc_subsample.power)
#         circ = plt.Circle((0, 0), self.singlemode_fiber_core_radius, edgecolor='white', fill=False, linewidth=2, alpha=0.5)
#         # circ = plt.Circle((0, 0), self.multimode_fiber_core_radius, edgecolor='white', fill=False, linewidth=2, alpha=0.5)
#         plt.gca().add_artist(circ)
#         plt.xlabel('x (um)')
#         plt.ylabel('y (um)')
#         plt.colorbar()
        
#         plt.show(block=False)
#         plt.pause(0.05)
#         plt.clf()

#         if self.timestep_render+1 == self.max_steps:
#             plt.close()


#     def parameters_init(self, act_dim, velocity_value, obs_dim, timesteps_per_episode, fried_parameter):
        
#         # Adjusting the velocity value
#         if (self.atm_type == 'quasi_static' or self.atm_type == 'semi_dynamic') and velocity_value != 0:
#             print('In ' + self.atm_type + ' atmospheric condition, the velocity value should be zero.')
#             print('therefore velocity value is changed to zero')
#             velocity_value = 0
        
#         elif self.atm_type == 'dynamic' and velocity_value == 0:
#             print('In ' + self.atm_type + ' atmospheric condition, the velocity value cannot be zero.')
#             print('therefore velocity value is changed to 1 m/s')
#             velocity_value = 1
            
#         # The parameters used for the simulation
#         parameters = {
#             # telescope configuration:
#             'telescope_diameter': 0.5,                     # diameter of the telescope in meters

#             # pupil configuration
#             'num_pupil_pixels' : 240,                      # Number of pupil grid pixels

#             # wavefront configuration
#             'wavelength_wfs' : 1.5e-6,                     # wavelength of wavefront sensing in micro-meters
#             'wavelength_sci' : 2.2e-6,                     # wavelength of scientific channel in micro-meters

#             # deformable mirror configuration
#             'num_modes' : act_dim+1,                         # Number of actuators in Deformable mirror

#             # Atmosphere configuration
#             # 'delta_t': 1e-3,                               # in seconds, for a loop speed of 1 kHz
#             'max_steps': timesteps_per_episode,                               # Maximum number of timesteps in an episode
#             'velocity': velocity_value,                    # the velocity of attmosphere
#             'fried_parameter' : fried_parameter,                      # The Fried parameter in meters
#             'outer_scale' : 10,                            # The outer scale of the phase structure function in meters

#             # Fiber configuration
#             'D_pupil_fiber' : 0.5,                         # Diameter of the pupil for fiber
#             'num_pupil_pixels_fiber': 128,                 # Number of pupil grid pixels for fiber
#             'num_focal_pixels_fiber' : 128,                # Number of focal grid pixels for fiber
#             'num_focal_pixels_fiber_subsample' : obs_dim,  # Number of focal grid pixels for fiber subsampled for quadrant photodetector
#             'multimode_fiber_core_radius' : 25 * 1e-6,     # the radius of the multi-mode fiber
#             'singlemode_fiber_core_radius' : 9 * 1e-6,     # the radius of the single-mode fiber
#             'fiber_NA' : 0.14,                             # Fiber numerical aperture 
#             'fiber_length': 10,

#             # wavefront sensor configuration
#             'f_number' : 50,                               # F-ratio
#             'num_lenslets' : 12,                           # Number of lenslets along one diameter
#             'sh_diameter' : 5e-3,                          # diameter of the sensor in meters
#             'stellar_magnitude' : -5,                      # measurement of brightness for stars and other objects in space
#             }

#         for param, val in parameters.items():
#             if isinstance(val, str) == False:
#                 exec('self.' + param + ' = ' + str(val))
    
    
#     def SH_step(self):
        
#         """
#         This section is to generate action by using Shack-Hartmann sensor
#         Since Shack-Hartmann uses information from the environment directly, \
#         the action needs to be generated in here.
#         """

#         # Propagatation of wavefront through atmosphere
#         wf_wfs_after_atmos = self.layer(self.wf_wfs)
        
#         for i in range(10):
#             # Propagatation of wavefront through deformable mirror
#             wf_wfs_after_dm = self.deformable_mirror_shack(wf_wfs_after_atmos)

#             # Propagatation of wavefront through Shack-Hartmann wavefront sensor
#             wf_wfs_on_sh = self.shwfs(self.magnifier(wf_wfs_after_dm))

#             # Read out WFS camera
#             self.camera.integrate(wf_wfs_on_sh, self.delta_t)
#             wfs_image = self.camera.read_out()
#             wfs_image = large_poisson(wfs_image).astype('float')

#             # calculate slopes from WFS image
#             slopes = self.shwfse.estimate([wfs_image + 1e-10])
#             slopes -= self.slopes_ref
#             slopes = slopes.ravel()

#             # generate the next action
#             gain = 0.3
#             leakage = 0.01
#             self.deformable_mirror_shack.actuators = (1 - leakage) * \
#                 self.deformable_mirror_shack.actuators - gain * self.reconstruction_matrix.dot(slopes)

#         action = self.deformable_mirror_shack.actuators

#         return action


#     def pupil_simulation(self):
#         """
#         simulating the pupil
#         Modeling the the diameter of the pupil of a telescope as a function of the telescope's diameter
#         """
        
#         pupil_grid_diameter = self.telescope_diameter
#         pupil_grid = make_pupil_grid(self.num_pupil_pixels, pupil_grid_diameter)
#         aperture = make_circular_aperture(self.telescope_diameter)(pupil_grid)
        
#         return aperture, pupil_grid 
        

#     def incoming_wavefront(self, aperture, pupil_grid):
#         """
#         Incoming wavefront
#         Simulation of the incoming wavefront from the satellite
#         """
        
#         # propagation of a wavefront through a perfect lens
#         spatial_resolution = self.wavelength_sci / self.telescope_diameter                               # The physical size of a resolution element
#         focal_grid = make_focal_grid(q=4, num_airy=30, spatial_resolution=spatial_resolution)  # Make a grid for a focal plane
        
#         self.propagator = FraunhoferPropagator(pupil_grid, focal_grid)
        
#         # created unaberrated point spread function (ideal propagation) for future comparison for Strehl ratio
#         wf = Wavefront(aperture, self.wavelength_sci)
#         wf.total_power = 1
#         self.unaberrated_PSF = self.propagator.forward(wf).power
        
#         # generate the wavefront sensing
#         zero_magnitude_flux = 3.9e10                                       #3.9e10 photon/s for a mag 0 star
#         self.wf_wfs = Wavefront(aperture, self.wavelength_wfs)
#         self.wf_wfs.total_power = zero_magnitude_flux * 10**(-self.stellar_magnitude / 2.5)
        
#         # generate the wavefront sensing with total power of 1 for fiber coupling
#         self.wf_wfs_fiber = Wavefront(aperture, self.wavelength_wfs)
#         self.wf_wfs_fiber.total_power = 1
        
#         # generate the wavefront of scientific channel
#         self.wf_sci= Wavefront(aperture, self.wavelength_sci)
#         self.wf_sci.total_power = zero_magnitude_flux * 10**(-self.stellar_magnitude / 2.5)
        
#         return focal_grid


#     def DM_function(self, act_type, pupil_grid):
#         """
#         Deformable mirror function
#         """
        
#         if act_type == 'zernike':
#             # generate the deformable mirror with zernike
#             dm_modes = make_zernike_basis(self.num_modes, self.telescope_diameter, pupil_grid)
#             dm_modes = ModeBasis([mode / np.ptp(mode) for mode in dm_modes], pupil_grid)
#             self.deformable_mirror = DeformableMirror(dm_modes)
            
#         else:
#             # generate the deformable mirror with number of actuators
#             dm_modes = make_disk_harmonic_basis(pupil_grid, self.num_modes, self.telescope_diameter, 'neumann')
#             dm_modes = ModeBasis([mode / np.ptp(mode) for mode in dm_modes], pupil_grid)
#             self.deformable_mirror = DeformableMirror(dm_modes)
            
#             self.deformable_mirror.flatten()
        
#         return dm_modes


#     def atmospheric_turbulence(self, pupil_grid):
#         """
#         Atmospheric Turbulence - Simulating the atmosphere
#         """
        
#         # Calculate the integrated Cn^2 for a certain Fried parameter
#         Cn_squared = Cn_squared_from_fried_parameter(self.fried_parameter, self.wavelength_sci)
        
#         # create the layer for the atmosphere
#         self.layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, self.outer_scale, self.velocity)
        
#         # with open('layer.pkl', 'wb') as f:
#         #     pickle.dump(self.layer, f)
        
#         # self.layer = pd.read_pickle('layer.pkl')

#         # layer_dir = './Layers/Layers'
#         # layer_name = 'layer_' + str(self._layer_no) + '.pkl'
#         # file_path = os.path.join(layer_dir, layer_name)
#         # self.layer = pd.read_pickle(file_path)
        
    
#     def fiber_coupling(self):
#         """
#         Fiber coupling function
#         """
        
#         pupil_grid_fiber = make_pupil_grid(self.num_pupil_pixels_fiber, self.D_pupil_fiber)
        
#         # The diameter of the grid for fiber
#         D_focus_fiber = 2.1 * self.multimode_fiber_core_radius
        
#         # make the grid for focal plane before fiber
#         focal_grid_fiber = make_pupil_grid(self.num_focal_pixels_fiber, D_focus_fiber)
#         focal_grid_fiber_subsample = make_pupil_grid(self.num_focal_pixels_fiber_subsample, D_focus_fiber)
        
#         # propagation of a wavefront through a focal plane before fiber
#         focal_length = self.D_pupil_fiber/(2 * self.fiber_NA)                        # The focal length of the lens system
        
#         self.propagator_fiber = FraunhoferPropagator(pupil_grid_fiber, focal_grid_fiber, focal_length=focal_length)
#         self.propagator_fiber_subsample = FraunhoferPropagator(pupil_grid_fiber, focal_grid_fiber_subsample, focal_length=focal_length)
        
#         self.single_mode_fiber = StepIndexFiber(self.singlemode_fiber_core_radius, self.fiber_NA, self.fiber_length)
        
    
#     def shack_hartmann_init(self, pupil_grid, focal_grid, aperture, dm_modes):
#         """
#         This part is for the initialization of the Shack Hartmann wavefront sensor
#         The diameter of the beam needs to be reshaped with a magnifier, otherwise ...  
#         the spots are not resolved by the pupil grid 
#         """
#         from hcipy import Magnifier, SquareShackHartmannWavefrontSensorOptics, \
#             ShackHartmannWavefrontSensorEstimator, NoiselessDetector, \
#             inverse_tikhonov
            
#         magnification = self.sh_diameter / self.telescope_diameter
#         self.magnifier = Magnifier(magnification)
        
#         # create the shack-hartmann wavefront sensor:
#         self.shwfs = SquareShackHartmannWavefrontSensorOptics(pupil_grid.scaled(magnification), \
#                                                               self.f_number, self.num_lenslets, self.sh_diameter)
#         self.shwfse = ShackHartmannWavefrontSensorEstimator(self.shwfs.mla_grid, self.shwfs.micro_lens_array.mla_index)
        
#         # create the noiseless detector for Shack-Hartmann
#         self.camera = NoiselessDetector(focal_grid)
#         wf_camera = Wavefront(aperture, self.wavelength_wfs)
#         self.camera.integrate(self.shwfs(self.magnifier(wf_camera)), 1)
#         image_ref = self.camera.read_out()
            
#         # select subapertures to use for wavefront sensing, based on their flux:
#         fluxes = ndimage.sum(image_ref, self.shwfse.mla_index, self.shwfse.estimation_subapertures)
#         flux_limit = fluxes.max() * 0.5
        
#         # generate the Shack-Hartmann wavefront sensor estimator:
#         estimation_subapertures = self.shwfs.mla_grid.zeros(dtype='bool')
#         estimation_subapertures[self.shwfse.estimation_subapertures[fluxes > flux_limit]] = True
        
#         self.shwfse = ShackHartmannWavefrontSensorEstimator(self.shwfs.mla_grid, self.shwfs.micro_lens_array.mla_index, estimation_subapertures)

#         # calculate reference slopes
#         self.slopes_ref = self.shwfse.estimate([image_ref])
        
#         # create a deformable mirror for shack-hartmann to prevent any confusion
#         self.deformable_mirror_shack = DeformableMirror(dm_modes)
        
#         # calibrating the interaction matrix:
#         probe_amp = 0.01 * self.wavelength_wfs
#         response_matrix = []

#         wf_cal = Wavefront(aperture, self.wavelength_wfs)
#         wf_cal.total_power = 1
        
#         for i in range(self.num_modes):
#             slope = 0

#             # Probe the phase response
#             amps = [-probe_amp, probe_amp]
#             for amp in amps:
#                 self.deformable_mirror_shack.flatten()
#                 self.deformable_mirror_shack.actuators[i] = amp

#                 dm_wf = self.deformable_mirror_shack.forward(wf_cal)
#                 wfs_wf = self.shwfs(self.magnifier(dm_wf))

#                 self.camera.integrate(wfs_wf, 1)
#                 image = self.camera.read_out()

#                 slopes = self.shwfse.estimate([image])

#                 slope += amp * slopes / np.var(amps)

#             response_matrix.append(slope.ravel())

#         response_matrix = ModeBasis(response_matrix)

#         # inversion of interaction matrix using Tikhonov regularization
#         rcond = 1e-3
#         self.reconstruction_matrix = inverse_tikhonov(response_matrix.transformation_matrix, rcond=rcond)
    
    
#     def reward_function(self, next_state):
        
#         # wavefront after passing through the single-mode fiber
#         wf_smf = self.single_mode_fiber.forward(self.wf_wfs_after_foc)
        
#         # total power of the wavefront after single-mode fiber
#         rew_fiber = wf_smf.total_power
            
#         if self.rew_type == 'strehl_ratio':
            
#             # Propagate the Near-Infrared wavefront
#             self.wf_sci_focal_plane = self.propagator(self.deformable_mirror(self.layer(self.wf_sci)))

#             # calculate the strehl ratio and the cost
#             strehl_ratio = get_strehl_from_focal(self.wf_sci_focal_plane.power, \
#                                                  self.unaberrated_PSF * self.wf_wfs.total_power) * 100
                
#             reward = - (100 - strehl_ratio)

#         elif self.rew_type == 'smf_ssim':
#             # calculation for ssim
#             focal_power = (self.wf_wfs_after_foc_subsample.power)

#             ref_power = np.zeros(self.num_focal_pixels_fiber_subsample**2)
#             ref_power[int(self.num_focal_pixels_fiber_subsample**2/2)] = self.obs_max   # this value needs to be optimized
            
#             data_range = ref_power.max() - ref_power.min()  # Calculate the data range for your images
#             ssim_score = ssim(focal_power, ref_power, data_range=data_range) 
            
#             # coef_rew = 0.8
#             tip_tilt_error = self.tip_tilt_calc(next_state)
#             # reward = rew_fiber/0.757 - (self._c_rew) * tip_tilt_error
            
#             reward = rew_fiber/0.757 - 0.05 * tip_tilt_error
            
#             # reward = rew_fiber/0.757 - self._c_rew * tip_tilt_error 
#             # reward = rew_fiber/0.757 + self._c_rew * ssim_score
            
#             # reward = rew_fiber/0.757 + self._c_rew * 0.1 * ssim_score - (1-self._c_rew) * 0.05 * tip_tilt_error 
            
#         if self.rew_threshold is not None and reward < self.rew_threshold:
#             reward = -1.0
        
#         return reward, rew_fiber


#     def tip_tilt_calc(self, next_state):
        
#         # calculating the tip and tilt
#         p = np.array([[20, 21, 22, 23, 24], 
#                      [15, 16, 17, 18, 19],
#                      [10, 11, 12, 13, 14],
#                      [5, 6, 7, 8, 9],
#                      [0, 1, 2, 3, 4]]
#                      )
        
#         pi1_pi2_sum = 0
#         pi4_pi5_sum = 0
#         p1j_p2j_sum = 0
#         p4j_p5j_sum = 0
        
#         for i in range(5):
#             pi1_pi2_sum += next_state[p[i,0]] + next_state[p[i,1]]
#             pi4_pi5_sum += next_state[p[i,3]] + next_state[p[i,4]]
            
#             p1j_p2j_sum += next_state[p[0,i]] + next_state[p[1,i]]
#             p4j_p5j_sum += next_state[p[3,i]] + next_state[p[4,i]]
        
#         norm_factor = np.sum(next_state)
        
#         tip_error = abs(pi1_pi2_sum - pi4_pi5_sum)/norm_factor # tip
#         tilt_error = abs(p1j_p2j_sum - p4j_p5j_sum)/norm_factor  # tilt

#         error_sum = tip_error + tilt_error
        
#         return error_sum
