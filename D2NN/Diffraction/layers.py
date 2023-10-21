import torch
from torch.nn import *

import numpy as np
from matplotlib.pyplot import *

class RSDiffraction(Module):
    def __init__(self, z, wlen, ds, input_shape, output_shape, n=1., verbose=0, types=(torch.float32, torch.complex64)):
        super(RSDiffraction, self).__init__()
        
        self.z = z
        self.k = n*2*np.pi/wlen
        self.wlen = wlen
        self.ds = ds
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.verbose = verbose
        self.types = types

        self.QualityFactor()
        
        self.InitiateLayer()
                        
    def NumericalPadding(self):
        self.numerical_padding = ZeroPad2d((0, self.output_shape[0]-1, 0, self.output_shape[1]-1))

    def DiffractionPadding(self):
        w_pad = int((self.output_shape[0]-self.input_shape[0])/2)
        h_pad = int((self.output_shape[1]-self.input_shape[1])/2)

        self.diffraction_padding = ZeroPad2d((w_pad, w_pad, h_pad, h_pad))

    def InitiateLayer(self):
        self.Kernel()  ## compute kernel
        self.SimpsonMatrix()  ## compute Simpson Matrix
        
        # Padding Layers
        self.DiffractionPadding()
        self.NumericalPadding()
        
    def QualityFactor(self):
        r_max = np.sqrt((int(self.output_shape[0]/2 -.5)*self.ds)**2 + (int(self.output_shape[1]/2)*self.ds)**2)
        ds_ideal = np.sqrt((self.wlen)**2 + r_max**2 + 2*self.wlen*np.sqrt(r_max**2 + self.z**2)) - r_max
        
        self.Q_factor = .5*ds_ideal/self.ds
        
        if self.verbose is 1:
            if self.Q_factor > 1.:
                print(f"Quality Factor Q={self.Q_factor}>1 indicates reliable simulation.")
            elif self.Q_factor <= 1.:
                print(f"Quality Factor Q={self.Q_factor}<1 indicates unreliable simulation.")

    def Kernel(self):
        psf = self.OnAxisPointSpreadFunction()
        self.kernel = torch.fft.fft2(psf, norm="backward")
        
        if self.verbose is 2:
            fig, axs = subplots(2, 2)

            axs[0,0].set_title(r"$\mathbb{Re}\left[psf\right]$")
            axs[0,1].set_title(r"$\mathbb{Im}\left[psf\right]$")
            axs[1,0].set_title(r"$\mathbb{Re}\left[kernel\right]$")
            axs[1,1].set_title(r"$\mathbb{Im}\left[kernel\right]$")
            
            ax = axs[0,0].imshow(psf.real)
            fig.colorbar(ax, ax=axs[0,0])
            ax = axs[0,1].imshow(psf.imag)
            fig.colorbar(ax, ax=axs[0,1])
            ax = axs[1,0].imshow(self.kernel.real)
            fig.colorbar(ax, ax=axs[1,0])
            ax = axs[1,1].imshow(self.kernel.imag)
            fig.colorbar(ax, ax=axs[1,1])
            
            axs[0,0].set_xticks([])
            axs[0,0].set_yticks([])
            axs[0,1].set_xticks([])
            axs[0,1].set_yticks([])
            axs[1,0].set_xticks([])
            axs[1,0].set_yticks([])
            axs[1,1].set_xticks([])
            axs[1,1].set_yticks([])

    def SimpsonMatrix(self):
        B = torch.zeros((1, self.input_shape[0]))
        B[0, ::2] = 4
        B[0, 1::2] = 2
        
        B[0, 0]=1.
        B[0, -1]=1
                
        B = B/3.
                
        self.W = torch.matmul(torch.transpose(B, 0, 1), B)
        
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

        real = torch.tensor([0], dtype=self.types[0])
        imag = torch.tensor([1], dtype=self.types[0])
        j = torch.complex(real, imag)

        psf = (torch.exp(j * k * r_abs) * z * ((1/r_abs) - j*k))/(2*np.pi*r_abs**2)

        return psf

    def OnAxisPointSpreadFunction(self):
        """Point Spread Function at the plane z centered around the axis.

        Args:
            z (float64): propagation distance
            k (float64): wavenumber
            ds (float64): sampling interval of the input plane
            computation_window_shape (tuple(int, int)): Pixel (width, height) of the output window.

        Returns:
            tensor(torch.complex128): Point Spread Function in the spatial domain in the plane at a propagation distance z.
        """

        x = torch.arange(-(self.output_shape[0]/2 -.5)*self.ds, (self.output_shape[0]/2)*self.ds, self.ds).type(self.types[1])
        y = torch.arange(-(self.output_shape[1]/2 -.5)*self.ds, (self.output_shape[1]/2)*self.ds, self.ds).type(self.types[1])
        
        X = torch.zeros(2*self.output_shape[0]-1)
        Y = torch.zeros(2*self.output_shape[1]-1)

        X[:self.output_shape[0]-1] = float(x[0]) - torch.flip(x[1:], dims=(0,))
        X[self.output_shape[0]-1:] = x-float(x[0])
        
        Y[:self.output_shape[1]-1] = float(y[0]) - torch.flip(y[1:], dims=(0,))
        Y[self.output_shape[1]-1:] = y-float(y[0])
        
        X, Y = torch.meshgrid(X, Y)

        psf = self.PointSpreadFunction(X, Y, self.z, self.k)

        return psf
    
    def Propagate(self, x):
        x = torch.mul(self.W, x)
        
        x = self.diffraction_padding(x)
        
        x = self.numerical_padding(x)
        
        x = torch.fft.fft2(x, norm="backward")

        output_fft = torch.mul(self.kernel, x)

        output = torch.fft.ifft2(output_fft, norm="backward")[:, -self.output_shape[0]:, -self.output_shape[1]:]
        
        return output*self.ds**2
    
class IntensityMeasurement(Module):
    def __init__(self):
        super(IntensityMeasurement, self).__init__()

    def Measure(self, x):
        return torch.abs(x)**2

