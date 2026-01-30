import numpy as np
import matplotlib.pyplot as plt


def numpy_zoom(arr, zoom_factor):
    """
    Approximate scipy.ndimage.zoom using only NumPy.
    Performs bilinear interpolation for 2D arrays.
    """
    if arr.ndim != 2:
        raise ValueError("This function currently supports only 2D arrays.")

    # Original coordinates
    h, w = arr.shape
    y = np.arange(h)
    x = np.arange(w)

    # Target size
    new_h = int(np.round(h * zoom_factor))
    new_w = int(np.round(w * zoom_factor))

    # New coordinates (normalized to original index space)
    y_new = np.linspace(0, h - 1, new_h)
    x_new = np.linspace(0, w - 1, new_w)
    x_new_grid, y_new_grid = np.meshgrid(x_new, y_new)

    # Floor and ceil coordinates
    x0 = np.floor(x_new_grid).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(y_new_grid).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)

    # Distances
    dx = x_new_grid - x0
    dy = y_new_grid - y0

    # Bilinear interpolation
    top = (1 - dx) * arr[y0, x0] + dx * arr[y0, x1]
    bottom = (1 - dx) * arr[y1, x0] + dx * arr[y1, x1]
    result = (1 - dy) * top + dy * bottom

    return result

def evaluate_wavefront(wavefront):
    """Returns the (scalar) L2 deviation w.r.t. a flat wavefront"""
    
    centered_wavefront = wavefront - np.mean(wavefront)
    return np.mean(centered_wavefront**2)

def create_circular_mask(num_points):
    x = np.linspace(-1, 1, num_points)
    y = np.linspace(-1, 1, num_points)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    mask = R <= 1

    return mask

def apply_circular_mask(surface):
    # create the circular mask
    num_points = surface.shape[0]
    mask = create_circular_mask(num_points)

    # mask out points outside the circle
    masked_surface = np.where(mask, surface, np.nan)

    return masked_surface


class MinimalAOSimulator:

    def __init__(self, **env_params):
        self.rng_seed = env_params['rng_seed']
        self.rng = np.random.default_rng(seed=self.rng_seed)
        self.noise_scale = env_params.get('noise_scale', 0.2)

        # create the deformable mirror
        self.mirror_size = env_params.get('square_mirror_size', 60)
        assert self.mirror_size % 2 == 0, "Mirror side should be a multiple of 4"
        self.mirror_shape = (self.mirror_size, self.mirror_size)

        self.low_res_size = env_params.get('low_res_size', (4,4))
        self.zoom_times = self.mirror_shape[0] // self.low_res_size[0]
        self.circular_mask = create_circular_mask(self.mirror_size)
        self.obs_discretization_param = env_params.get('obs_discretization_param', 5)   # default 5x5 discretization
        assert self.mirror_size % self.obs_discretization_param == 0, 'self.mirror_shape should be divisible by self.obs_discretization_param'
        self.obs_discretization_block_size = self.mirror_size // self.obs_discretization_param

        # create the modes
        self.modes = self._create_modes()

    def _create_modes(self):
        """Returns the first 10 modes of the Zernike polynomial"""
        
        domain_x = np.linspace(-1, 1, self.mirror_shape[0])
        domain_y = np.linspace(-1, 1, self.mirror_shape[1])

        # zeroth-order mode
        z_0_0 = np.ones(self.mirror_shape)

        # first-order modes
        z_1_1m = np.zeros(self.mirror_shape)
        z_1_1 = np.zeros(self.mirror_shape)
        for x in range(self.mirror_shape[0]):
            for y in range(self.mirror_shape[1]):
                z_1_1m[x,y] = domain_y[y]
                z_1_1[x,y] = domain_x[x]
        
        # second-order modes
        z_2_2m = np.zeros(self.mirror_shape)
        z_2_0 = np.zeros(self.mirror_shape)
        z_2_2 = np.zeros(self.mirror_shape)
        for x in range(self.mirror_shape[0]):
            for y in range(self.mirror_shape[1]):
                z_2_2m[x,y] = 2 * domain_x[x] * domain_y[y]
                z_2_0[x,y] = 2 * domain_x[x]**2 + 2 * domain_y[y]**2 - 1
                z_2_2[x,y] = -domain_x[x]**2 + domain_y[y]**2        

        # third-order modes
        z_3_3m = np.zeros(self.mirror_shape)
        z_3_1m = np.zeros(self.mirror_shape)
        z_3_1 = np.zeros(self.mirror_shape)
        z_3_3 = np.zeros(self.mirror_shape)
        for x in range(self.mirror_shape[0]):
            for y in range(self.mirror_shape[1]):
                z_3_3m[x,y] = -domain_x[x]**3 + 3 * domain_x[x] * domain_y[y]**2
                z_3_1m[x,y] = -2 * domain_x[x] + 3 * domain_x[x]**3  + 3 * domain_x[x] * domain_y[y]**2
                z_3_1[x,y] = -2 * domain_y[y] + 3 * domain_y[y]**3  + 3 * domain_x[x]**2 * domain_y[y]
                z_3_3[x,y] = domain_y[y]**3 - 3 * domain_x[x]**2 * domain_y[y]

        return [z_0_0, z_1_1m, z_1_1, z_2_2m, z_2_0, z_2_2, z_3_3m, z_3_1m, z_3_1, z_3_3]
    
    def reset(self):
        # create a flat mirror
        self.mirror = np.zeros(self.mirror_shape)

        # reset the zernike coefficients
        self.zernike_coeffs = np.zeros(len(self.modes))

        # initialize a flat wavefront
        self.wavefront_size = self.mirror_size                  # ToDo: can be made more flexible
        self.wavefront = np.zeros(self.wavefront_size)

        # create some noise and add it to the wavefront as atmospheric distortion 
        self.wavefront = np.clip(self.wavefront + self._create_atmospheric_distortion(), -1.0, 1.0)

    def _create_atmospheric_distortion(self):
        """Create noise in [-1, 1] * noise_scale"""
        atmospheric_distortion_seed = (self.rng.integers(-2, 3, self.low_res_size)) * 0.5 * self.noise_scale  
        return numpy_zoom(atmospheric_distortion_seed, self.zoom_times)

    def start(self):
        """Simulates the reflection of the initial wavefront and return the resulting wavefront"""
        self.reset()
        self.reflected_wavefront = self.mirror - self.wavefront
        
        return self._aggregate_data(self.reflected_wavefront)

    def step(self, action):
        """
        Simulates a step of atmospheric distortion of the wavefront, 
        computes its reflection (next observation) based on the current zernike modes (action), 
        evaluates the quality of the reflected wavefront (reward),
        returns the next observation and reward.
        """
        # create the new shape of the mirror
        self.mirror = self._create_mirror_from_zernike_coeffs(action)

        # simulate the reflection
        self.reflected_wavefront = self.mirror - self.wavefront
        observation = self._aggregate_data(self.reflected_wavefront)

        # evaluate the reflected wavefront
        reward = -evaluate_wavefront(self.reflected_wavefront)

        # create some noise and add it to the wavefront as atmospheric distortion 
        self.wavefront = np.clip(self.wavefront + self._create_atmospheric_distortion(), -1.0, 1.0)

        return reward, observation

    def _create_mirror_from_zernike_coeffs(self, zernike_coeffs):
        mirror = np.zeros(self.mirror_shape)
        for i in range(len(self.modes)):
            mirror += zernike_coeffs[i] * self.modes[i]
        mirror_clipped = np.clip(mirror, -1.0, 1.0)

        return mirror_clipped
    
    def _aggregate_data(self, data):
        """Aggregates a masked array into square blocks and returns a 1D-version of the result"""
        h, w = self.mirror.shape
        new_h, new_w = h // self.obs_discretization_block_size, w // self.obs_discretization_block_size
        
        # Crop to fit the block size and reshape to 4D
        data_view = data[:new_h*self.obs_discretization_block_size, :new_w*self.obs_discretization_block_size].reshape(new_h, self.obs_discretization_block_size, new_w, self.obs_discretization_block_size)
        mask_view = self.circular_mask[:new_h*self.obs_discretization_block_size, :new_w*self.obs_discretization_block_size].reshape(new_h, self.obs_discretization_block_size, new_w, self.obs_discretization_block_size)
        
        # Only aggregate the data inside the mask
        valid_sum = np.sum(data_view * mask_view, axis=(1, 3))
        valid_count = np.sum(mask_view, axis=(1, 3))            # ToDo: this can be computed once and stored
        
        # Handle division by zero for fully masked blocks
        with np.errstate(divide='ignore', invalid='ignore'):
            result = valid_sum / valid_count

        # Return a flatten version of the result while ignoring the cells without enough data
        return result[valid_count > 0.5 * np.max(valid_count)]

    def render(self):
         
        raw_data = []
        size = None

        surfaces = [self.wavefront, self.mirror, self.reflected_wavefront]
        cmaps = ['viridis', 'plasma', 'viridis']
        titles = ['Incident Wavefront', 'Mirror', 'Reflected Wavefront']

        for i in range(3):

            # if i==0:
                # vmax = np.max(surfaces[0])
                # vmin = np.min(surfaces[0])
            fig, ax = plt.subplots(figsize=[4, 4])
            image = ax.imshow(apply_circular_mask(surfaces[i]), cmap=cmaps[i], vmin=-1.0, vmax=1.0,
                      origin='lower', animated=True)
            plt.colorbar(image)
            ax.set_title(titles[i])
            ax.set_axis_off()
            canvas = fig.canvas
            canvas.draw()
            raw_data.append(canvas.buffer_rgba())
            size = canvas.get_width_height()
            plt.close(fig)

        # mirror = ax1.imshow(apply_circular_mask(self.mirror), cmap='plasma', origin='lower', animated=True)
        # wavefront_reflected = ax2.imshow(apply_circular_mask(self.reflected_wavefront), cmap='viridis', origin='lower', animated=True)
        # ax0.set_title('Incident wavefront')
        # ax1.set_title('Mirror')
        # ax2.set_title('Reflected wavefront')
        # ax0.set_axis_off(); ax1.set_axis_off(); ax2.set_axis_off()

        


        # return (wavefront_incoming, mirror, wavefront_reflected)
        return tuple(raw_data), size
