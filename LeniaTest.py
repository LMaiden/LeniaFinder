################################################ Library imports ######################################################################################################

import numpy as np
import scipy.signal
import matplotlib.pylab as plt
import matplotlib.animation
import IPython.display

################################################ Custom imports #############################################################################################################

import Functions as F
import Settings as S
#__________________________________________________ Simulation parameters ________________________________________________________________________

size =int(S.size*2) ;  mid = S.mid;  scale = S.scale;  cx, cy = S.cx, S.cy
frame_id = 0
#________________________________________________________________________________________________________________________________________________

def run_world_execute(me, show_animation, max_iterations=2000):
  

  #Load Entity
  globals().update(me)
  
  R = me['R']
  T = me['T']
  kernels = me['kernels']
  cells = me['cells']

  #Place scaled entity
  global As, img
  As = [ np.zeros([size, size]) for i in range(3) ]
  Cs = [ scipy.ndimage.zoom(np.asarray(c), scale, order=0) for c in cells ];  R *= scale
  for A,C in zip(As,Cs):  A[cx:cx+C.shape[0], cy:cy+C.shape[1]] = C

  #Multiple kernels preparation
  Ds = []
  for k in kernels:
    y, x = np.ogrid[:size, :size]
    y = y - size // 2
    x = x - size // 2
    D = (np.sqrt(x**2 + y**2)) / R * len(k['b']) / k['r']
    Ds.append(D)
      
  Ks = [ (D<len(k['b'])) * np.asarray(k['b'])[np.minimum(D.astype(int),len(k['b'])-1)] * F.bell(D%1, 0.5, 0.15) for D,k in zip(Ds,kernels) ]
  nKs = [ K / np.sum(K) for K in Ks ]
  fKs = [ np.fft.fft2(np.fft.fftshift(K)) for K in nKs ]


  def growth(U, m , s):
      return F.bell(U, m, s)*2-1

  def update(i):
    
    global As, img
    ''' calculate convolution from source channels c0 '''
    fAs = [ np.fft.fft2(A) for A in As ]
    Us = [ np.real(np.fft.ifft2(fK * fAs[k['c0']])) for fK,k in zip(fKs,kernels) ]
    
    ''' calculate growth values for destination channels c1 '''
    Gs = [ growth(U, k['m'], k['s']) for U,k in zip(Us,kernels) ]
    Hs = [ sum(k['h']*G for G,k in zip(Gs,kernels) if k['c1']==c1) for c1 in range(3) ]
    
    ''' add growth values to channels '''
    As = [ np.clip(A + 1/T * H, 0, 1) for A,H in zip(As,Hs) ]
    
    F.Save_tojson_array(As, f"data_{me['name']}.json", r'C:\Users\gweno\Documents\Homework\ResearchEC\data')
    # E.Eval_velocity(As)
    ####
    ''' show image in RGB '''
    img.set_array(np.dstack(As))
    return img,


  ############################################################# Animation setup #####################################################################################


  fig, ax = plt.subplots()
  img = ax.imshow(A, cmap='viridis', interpolation='nearest')

  ani = matplotlib.animation.FuncAnimation(
      fig, update, frames=200, interval=20, blit=True, repeat=False,
  )
  
  
  if show_animation:
    plt.show()
  plt.pause(4)
  plt.close()