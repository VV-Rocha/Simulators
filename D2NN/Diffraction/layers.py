import torch
from torch.nn import *

import numpy as np

class RSDiffraction(Module):
    def __init__(self, z, k, wlen, ds, shape, padding):
        super(RSDiffraction, self).__init__()

        self.ds = ds

        self.kernel = torch.fft.fft2(self.OnAxisPointSpreadFunction(z, k, ds, shape), norm="backward")
        self.kernel = torch.fft.fftshift(self.kernel)
        #self.kernel = AS(z, k, wlen, ds, shape)
        
        self.padding = padding

    def PointSpreadFunction(self, x, y, z, k):
        """Point Spread Function in the spatial domain.

        Args:
            x (float64): x position
            y (float64): y position
            z (float64): z position
            k (float64): wavenumber

        Returns:
            tensor(torch.complex128): Returns the Point Spread Function in position (x,y,z).
        """
        r_abs = torch.sqrt(x**2 + y**2 + z**2)

        real = torch.tensor([0], dtype=torch.float64)
        imag = torch.tensor([1], dtype=torch.float64)
        j = torch.complex(real, imag)

        psf = torch.exp(j * k * r_abs) * z * (1/r_abs - j*k)/(2*np.pi*r_abs**2)

        return psf

    def OnAxisPointSpreadFunction(self, z, k, ds, computation_window_shape):
        """Point Spread Function at the plane z centered around the axis.

        Args:
            z (float64): propagation distance
            k (float64): wavenumber
            ds (float64): sampling interval of the input plane
            computation_window_shape (tuple(int, int)): Pixel (width, height) of the output window.

        Returns:
            tensor(torch.complex128): Point Spread Function in the spatial domain in the plane at a propagation distance z.
        """
        xx = torch.arange(-int(computation_window_shape[0]/2)*ds, (1+int(computation_window_shape[0]/2))*ds, ds)
        yy = torch.arange(-int(computation_window_shape[1]/2)*ds, (1+int(computation_window_shape[1]/2))*ds, ds)

        xx = (xx[1:]+xx[:-1])/2
        yy = (yy[1:]+yy[:-1])/2

        xx, yy = torch.meshgrid(xx, yy)
        
        psf = self.PointSpreadFunction(xx, yy, z, k)

        return psf
        
    def Propagate(self, x):
        x = torch.fft.fft2(x, norm="backward")
        x = torch.fft.fftshift(x)

        output_fft = torch.mul(self.kernel, x)

        output = torch.fft.ifft2(output_fft, norm="backward")
        output = torch.fft.ifftshift(output)
        return output*self.ds**2
    
class IntensityMeasurement(Module):
    def __init__(self):
        super(IntensityMeasurement, self).__init__()

    def Measure(self, x):
        return torch.abs(x)**2
