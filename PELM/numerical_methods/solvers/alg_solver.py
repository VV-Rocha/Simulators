import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def CircleMask(r: float= 50, shape: Tuple[int,int]=(1024, 768), dtype: type=np.uint8) -> np.ndarray:
    """Create a circular filter. The speckle size depends inversely on the diameter of the circle.

    Args:
        r (float, optional): Radius of the circular mask. Defaults to 50.
        shape (Tuple[int,int], optional): Shape of the mask. Defaults to (1024, 768).
        dtype (type, optional): Type to be used for the circular mask values. Defaults to np.uint8.

    Returns:
        np.ndarray: Mask of the active circular region.
    """   
    canvas = np.zeros((1, *shape), dtype=dtype)
    x_mid, y_mid = int((shape[0]/2)), int((shape[1]/2))

    # create the circle
    circ = np.zeros((2*r,2*r), dtype=dtype)

    x = np.arange(2*r) - int((r-1))

    xx = np.array(np.meshgrid(x,x))
    xx = np.sum(xx**2, axis=0)

    circ[np.where(xx<=r**2)] = 1

    # place circle on mask 
    if r%2==0:
        canvas[0, x_mid-int(r):x_mid+int(r), y_mid-int(r):y_mid+int(r)] = circ
    else:
        canvas[0, x_mid-int(r):x_mid+int(r), y_mid-int(r):y_mid+int(r)] = circ

    canvas = np.swapaxes(canvas, axis1=1, axis2=2)

    return canvas

class PELM_algsolver():
    """Instance of the simulator exploring the algebraic model behind the photonic extreme learning machine framework containing solely linear propagations.
    
    Attributes:
        shape (Tuple[int,int], optional): Shape of the simulation box. Defaults to (400,400).
        filter_radius (float, optional): Radius of the circular filter applied to the input. Defaults to 110.
        dtype (type, optional): Type used for the arrays. Defaults to np.complex64.    
    """    
    def __init__(self,
                 shape: Tuple[int,int]=(400,400),
                 filter_radius: float=110,
                 dtype: type=np.complex64):
        """The constructor for PELM_algsolver.

        Args:
            shape (Tuple[int,int], optional): Shape of the simulation box. Defaults to (400,400).
            filter_radius (float, optional): Radius of the circular filter applied to the input. Defaults to 110.
            dtype (type, optional): Type used for the arrays. Defaults to np.complex64.
        """
        self.shape = shape
        self.filter_radius = filter_radius
        
        self.dtype = dtype
        
        ## Initialize Filter Mask
        self.Initiatefilter()
        
        ## Initialize Hidden Layer Random Weights
        self.Initialize()
        
    def Initiatefilter(self,):
        """Initiate the input filter circular mask."""
        self.filter = CircleMask(r = self.filter_radius, shape=self.shape, dtype=np.bool)[0]
        
    def Initialize(self,):
        """Initiate the random complex values acting as diffractive medium."""
        magnitudes = np.random.uniform(0.,1., self.shape).astype(dtype=np.float32)
        phases = np.random.uniform(-np.pi,np.pi, self.shape)
        self.hweights = (np.exp(1j*phases) * magnitudes).astype(dtype=self.dtype)
        
    def linear_mapping(self, X: NDArray[np.complex64]) -> NDArray[np.complex64]:
        """Solves the algebraic model by a random transmission matrix.

        Args:
            X (NDArray[np.complex64]): A (n_samples)x(width)x(height) numpy array containing the input information. The number precision is np.complex64 by default but can be changed by redefining argument dtype.

        Returns:
            NDArray[np.complex64]: A (n_samples)x(width)x(height) numpy array containing the electric fields resulting in the speckle patterns for each input sample. The number precision is np.complex64 by default but can be changed by redefining argument dtype.
        """        
        X = X * self.filter.astype(int)
        
        X = X * self.hweights

        fft2 = np.fft.fft2(X, axes=(1,2)).astype(self.dtype)
        fft2 = np.fft.fftshift(fft2, axes=(1,2))
        
        return fft2

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

    def solver(self, X: NDArray[np.complex64], normed=True, noise=None, measure=True) -> NDArray[np.float32]:
        """Simulates the linear transmission matrix and the intensity measurement.

        Args:
            X (NDArray[np.complex64]): A (n_samples)x(width)x(height) numpy array containing the input information. The number precision is np.complex64 by default but can be changed by redefining argument dtype.
            
        Returns:
            NDArray[np.float32]: Pixel Intensities.
        """
        
        output_fields = self.linear_mapping(X)
        
        if measure is True:    
            if noise is not None:
                output_fields = self.output_noise(output_fields, noise)
            else:
                output_fields = np.abs(output_fields)**2
            
            if normed is True:
                output_fields /= output_fields.max()
                
        return output_fields
        