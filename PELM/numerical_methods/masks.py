import numpy as np

def MacroMask(f, px, py):
    Np = int(px*py*f)
    N_row = int(Np/px)
    N_res = Np - N_row*px
    
    canvas = np.zeros((px,py))
    for _i in range(N_row):
        canvas[_i] = canvas[_i] + 1.
        
    if N_res != 0:
        canvas[N_row, :N_res] += 1. 
        
    return canvas
    
def FMask(NMx, NMy, Mmask):
    px, py = Mmask.shape
    
    canvas = np.zeros((px*NMx, py*NMy))
    for _i in range(NMx):
        for _j in range(NMy):
            canvas[_i*px:(_i+1)*px, _j*py:(_j+1)*py] = Mmask
    return canvas

def DMDEncoding(f1, f2, DMD_enc_w, DMD_enc_h, px,py):
    masks = np.zeros((2*DMD_enc_w, DMD_enc_h))
    
    #nx_repeat, ny_repeat = 1, 1 #int(DMD_enc_w/px), int(DMD_enc_h/py)
    
    NMx, NMy = int(DMD_enc_w/px), int(DMD_enc_h/py)
    
    mask_f1 = MacroMask(f1, int(px), int(py)) #np.repeat(np.repeat(MacroMask(f1, int(px), int(py)), nx_repeat, axis=0), ny_repeat, axis=1)
    
    mask_f2 = MacroMask(f2, int(px), int(py)) #np.repeat(np.repeat(MacroMask(f2, int(px), int(py)), nx_repeat, axis=0), ny_repeat, axis=1)
    
    masks[:DMD_enc_w] = FMask(NMx, NMy, mask_f1)
    masks[DMD_enc_w:] = FMask(NMx, NMy, mask_f2)
    
    return masks


def TwoFEncodings(feature_points,
                  indices,
                  mpx=(24,24),
                  encoding_shape=(240,240),
                  DMD_shape=(1024, 768),
                  ref_stride=2,
                  shift=(0,0)
                  ):
    ## % TODO: Add a method to shift the mask
    
    ___, M = feature_points.shape
    
    # Horizontal window
    x2 = int(DMD_shape[0]/2) + encoding_shape[0]
    x1 = int(DMD_shape[0]/2) - encoding_shape[0]
    y1 = int((DMD_shape[1] - encoding_shape[1])/2)
    y2 = int((DMD_shape[1] + encoding_shape[1])/2)

    # Vertical window
    x2 = int((DMD_shape[0] + encoding_shape[0])/2)
    x1 = int((DMD_shape[0] - encoding_shape[0])/2)
    y1 = int((DMD_shape[1])/2) - encoding_shape[1]
    y2 = int((DMD_shape[1])/2) + encoding_shape[1]

    DMDmask = np.zeros((M, *DMD_shape), dtype=np.uint8)
    DMDmask[:, ::ref_stride, ::ref_stride] += 1
    
    for m in range(M):
        encoding = DMDEncoding(feature_points[0, m], feature_points[1, m], *encoding_shape, *mpx).flatten()
        encoding = encoding[indices]
        encoding = np.reshape(encoding, (2*encoding_shape[0], encoding_shape[1]))
            
        DMDmask[m, x1:x2, y1:y2] = encoding.T
        
    DMDmask = np.swapaxes(DMDmask, 1, 2)
    
    DMDmask = np.roll(DMDmask, shift[0], axis=1)
    DMDmask = np.roll(DMDmask, shift[1], axis=2)
    
    return DMDmask

def ContinuosFeature(f1, encoding_shape, shape, noise=None, ref=1/3):
    M = f1.shape[0]
    masks = np.zeros((M, *shape), dtype=np.float32)
    
    x1 = int(shape[0]/2) - int(encoding_shape[0]/2)
    x2 = int(shape[0]/2) + int(encoding_shape[0]/2)
    
    y1 = int(shape[1]/2) - int(encoding_shape[1]/2)
    y2 = int(shape[1]/2) + int(encoding_shape[1]/2)
    
    masks += ref
    
    for _m in range(M):
        masks[_m, x1:x2, y1:y2] = np.full(encoding_shape, f1[_m])

    if noise is not None:
        masks += np.random.normal(0, noise/100, masks.shape)
        masks[np.where(masks<0)] = 0
        masks[np.where(masks>1)] = 1

    return masks

def ContinuosFeatures(f1, f2, encoding_shape, shape, ref=1/3):
    M = f1.shape[0]
    masks = np.zeros((M*M, *shape), dtype=np.float32)
    
    # f1 window
    x1 = int(shape[0]/2) - int(encoding_shape[0]/2)
    x2 = int(shape[0]/2) + int(encoding_shape[0]/2)
    y1 = int(shape[1]/2) - encoding_shape[1]
    y2 = int(shape[1]/2)
    
    # f2 window
    _x1 = int(shape[0]/2) - int(encoding_shape[0]/2)
    _x2 = int(shape[0]/2) + int(encoding_shape[0]/2)
    _y1 = int(shape[1]/2)
    _y2 = int(shape[1]/2) + encoding_shape[1]

    masks = masks + ref
    
    for m in range(M):
        for n in range(M):
            masks[n*M + m, x1:x2, y1:y2] = np.full(encoding_shape, f1[m])
            masks[m + n*M, _x1:_x2, _y1:_y2] = np.full(encoding_shape, f2[n])

    return np.swapaxes(masks, 1,2)