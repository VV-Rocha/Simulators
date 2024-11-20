import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from matplotlib.pyplot import *

from diffractio import um, nm, mm
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_fields_XY import Scalar_field_XY

## % TODO: The PELM_diffractioSolver seems to have a problem in the dimensional units of the grid resulting in a poor definition of the propagation distance. Consequently, the speckle pattern outputted lookes to be propagating more than expected.

class PELM_diffractioSolver():
    def __init__(self,
                 z = .1*mm,
                 filter_radius=.3*mm,
                 size = .1*mm,
                 wavelength=532*nm,
                 N=1024,
                 dtype=np.complex64,
                 random_amplitude=True):
        self.z = z
        self.filter_radius = filter_radius
        self.size = size
        self.wavelength = wavelength
        self.k = 2*np.pi/wavelength
        self.N = N
        
        self.dtype = dtype
        
        
        self.initiateMedium(random_amplitude)
        
        self.initiateGrid()
        
        self.Initiatefilter()

        self.initiatePropKernel()
        
        self.initiateField()

    def Initiatefilter(self):
        """Initiate the input filter circular mask."""
        self.filter = np.zeros((self.N, self.N), dtype=np.uint8)
    
        self.filter[np.where(np.sqrt(self.X**2 + self.Y**2)<=self.filter_radius)] = 1
    
    def plotSpeckle(self, input: NDArray[np.complex64]=1.) -> None:
        """Plots the output speckle intensity pattern and phase profile.

        Args:
            input (NDArray[np.complex64], optional): Input beam profile. A constant planewave can be introduced instead by a scalar complex value. Defaults to 1..
        """
        fig, axs = subplots(1)   
        speckle = np.abs(self.propagate(self.filter))**2
        axs.imshow(speckle, vmin=0, vmax=speckle.max(), extent=[-int(self.size/2), int(self.size/2), -int(self.size/2), int(self.size/2)])

        axs.set_xlabel(r"Width ($\mathbf{\mu m}$)", weight="bold")
        axs.set_ylabel(r"Height ($\mathbf{\mu m}$)", weight="bold")
    
    def initiateMedium(self, random_amplitude: bool=True) -> None:
        """Initiate the randomized amplitude and phase values representing the unpredictable 'surface' modulation of the input profile.

        Args:
            random_amplitude (bool, optional): Boolean value to turn on (True) or off (False) a randomization of the amplitude modulation. Defaults to True.
        """
        if random_amplitude == True:
            self.random_amplitude = np.random.uniform(0, 1, (self.N, self.N))
        else:
            self.random_amplitude = np.zeros((self.N,self.N)) + 1
        # Generate random phase screen
        self.random_phase = np.exp(1j * np.random.uniform(-np.pi, np.pi, (self.N, self.N))).astype(self.dtype)
        
    def initiateGrid(self) -> None:
        """Initiate the dimensional grid of discretized points."""
        self.x = np.linspace(-self.size/2, self.size/2, self.N, endpoint=False)  # Use endpoint=False to avoid double-counting the boundaries
        self.y = np.linspace(-self.size/2, self.size/2, self.N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def initiateField(self) -> None:
        """Initiate the scalar field for propagation."""
        # Create a source with the initial field and random phase
        u = Scalar_source_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        
        # Convert to Scalar_field_XY for propagation
        self.field = Scalar_field_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        
    def initiatePropKernel(self) -> None:
        """Initiate the Fourier Kernel for propagation of a beam a distance z."""
        # Create angular spectrum phase factor
        self.fx = np.fft.fftshift(np.fft.fftfreq(self.N, d=(self.x[1] - self.x[0])))
        self.fy = np.fft.fftshift(np.fft.fftfreq(self.N, d=(self.y[1] - self.y[0])))
        FX, FY = np.meshgrid(self.fx, self.fy)
        self.H = np.exp(1j * self.k * (FX**2 + FY**2) * self.z)
        
    def propagate(self, beam_profile: NDArray[np.complex64]) -> NDArray[np.complex64]:
        """Propagates the input spatial profiles.

        Args:
            beam_profile (NDArray[np.complex64]): Input spatial profile array with shape [nsamples, N,N].

        Returns:
            NDArray[np.complex64]: Output spatial profile array with shape [nsample, N,N].
        """
        self.field.u = beam_profile * self.filter
                
        beam_profile = np.multiply(self.field.u, self.random_amplitude*self.random_phase)
        
        # Perform Fourier transform to frequency domain
        U = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(beam_profile)))

        # Apply phase factor in frequency domain
        U_propagated = U * self.H

        # Inverse Fourier transform back to spatial domain
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_propagated)))
    
    def batch_propagation(self, X: NDArray[np.complex64]) -> NDArray[np.complex64]:
        """Simulates the propagation of a series of input samples.

        Args:
            X (NDArray[np.complex64]): Input sample images with shape (N_samples, width, height)

        Returns:
            NDArray[np.complex64]: output electric fields
        """
        N_samples, width, height = X.shape
     
        output_fields = np.zeros(X.shape, dtype=np.complex64)
        for _n in range(N_samples):
            output_fields[_n] = self.propagate(X[_n])
     
        return output_fields

    def output_noise(self, fields: NDArray[np.complex64], noise: float) -> NDArray[np.float32]:
        """Introduces noise in the measurement of the intensity as (|E|exp[ie] + a exp[ib]) -> |E|^2 + 2 a|E|cos(e-b) + |a|^2 where a and b are the amplitude and phase perturbations.

        Args:
            fields (NDArray[np.float32]): Electric field distributions.
            noise (float): Noise percentage of the total variation of the amplitude and phase.

        Returns:
            NDArray[np.float32]: Noisy intensity measurements.
        """
        
        E = np.abs(fields)
        
        a = np.random.normal(0, noise*np.max(E)/100, E.shape)
        
        b = np.random.normal(0, 2*np.pi*noise/100, E.shape)
        
        return E**2 + 2*a*E*np.cos(np.angle(fields) - b) + np.abs(a)**2

    def solver(self, X: NDArray[np.complex64], normed =True, noise=None) -> NDArray[np.float32]:
        """Simulation of a batch of input samples.

        Args:
            X (NDArray[np.complex64]): Input samples.
            normed (bool, optional): Whether output intensity is normalized between [0,1]. Defaults to True.
            noise (_type_, optional): Percentage of noise on the output. Defaults to None.

        Returns:
            NDArray[np.float32]:: Set of output intensities.
        """
        output_fields = self.batch_propagation(X)
        
        if noise is not None:
            output_fields = self.output_noise(output_fields, noise)
        else:
            output_fields = np.abs(output_fields)**2
        
        if normed is True:
            output_fields /= output_fields.max()
        
        return output_fields
    
    def plot(self, beam_profile=1.):
        try:
            if (beam_profile.shape != (self.N, self.N)):
                return None
        except AttributeError:
            pass

        if type(beam_profile) not in [int, float, np.float32, np.float64, np.complex64, np.complex128]:
            return None
        
        field = self.propagate(beam_profile)
        intensity = np.abs(field)**2
        
        fig, axs = subplots(1, 2, figsize=(8,4))
        axs[0].imshow(intensity, vmin=0, vmax=np.max(intensity), extent=(-self.size/2/um, self.size/2/um, -self.size/2/um, self.size/2/um))
        axs[0].set_title('Speckle Pattern')
        axs[0].set_xlabel('x (μm)')
        axs[0].set_ylabel('y (μm)')
        #axs[0].colorbar(label='Intensity')
        
        phase_profile = np.angle(field)
        
        axs[1].imshow(phase_profile, vmin=np.min(phase_profile), vmax=np.max(phase_profile), extent=(-self.size/2/um, self.size/2/um, -self.size/2/um, self.size/2/um), cmap='gray')
        axs[1].set_title('Phase Profile')
        axs[1].set_xlabel('x (μm)')
        axs[1].set_ylabel('y (μm)')
        #axs[1].colorbar(label='Phase')
