'''Lenia framework'''

###################################################################################
''' imports '''
import numpy as np
import scipy.signal
import matplotlib.pylab as plt
import matplotlib.animation
import IPython.display
import gc
###################################################################################
''' custom imports '''
import Lenia.lenia_settings as config
import Lenia.Functions.Math as math
import Lenia.lenia_data_manager as manager

import Lenia.EExamples.aquarium as aquarium

#__________________________________________________________________________________________________________________________

''' Global variables  '''
size = config.dim_XY
c = config.dim_C

cx = config.c_X
cy = config.c_Y

mid = config.mid

scale = config.scale

#___________________________________________________________________________________________________________________________

''' Lenia function '''

def run_world_execute(me, show_animation = False, max_iterations=2000):

  
  R = me['R']
  T = me['T']
  kernels = me['kernels']
  cells = me['cells']

  As = [ np.zeros([size, size]) for i in range(c) ]
  Cs = [ scipy.ndimage.zoom(np.asarray(c), scale, order=0) for c in cells ];  R *= scale
  for A,C in zip(As,Cs):  A[cx:cx+C.shape[0], cy:cy+C.shape[1]    ] = C


  Ds = []
  for k in kernels:
      y, x = np.ogrid[-mid:mid, -mid:mid]
      D = (np.sqrt(x**2 + y**2)) / R * len(k['b']) / k['r']
      Ds.append(D)
  Ks = [ (D<len(k['b'])) * np.asarray(k['b'])[np.minimum(D.astype(int),len(k['b'])-1)] * math.bell(D%1, 0.5, 0.15) for D,k in zip(Ds,kernels) ]
  nKs = [ K / np.sum(K) for K in Ks ]
  fKs = [ np.fft.fft2(np.fft.fftshift(K)) for K in nKs ]


  def growth(U, m , s):
      return math.bell(U, m, s)*2-1

  def update(i):
    
    nonlocal As, img
    ''' calculate convolution from source channels c0 '''
    fAs = [ np.fft.fft2(A) for A in As ]
    Us = [ np.real(np.fft.ifft2(fK * fAs[k['c0']])) for fK,k in zip(fKs,kernels) ]
    
    ''' calculate growth values for destination channels c1 '''
    Gs = [ growth(U, k['m'], k['s']) for U,k in zip(Us,kernels) ]
    Hs = [ sum(k['h']*G for G,k in zip(Gs,kernels) if k['c1']==c1) for c1 in range(c) ]
    
    ''' add growth values to channels '''
    As = [ np.clip(A + 1/T * H, 0, 1) for A,H in zip(As,Hs) ]
    
    ''' save current frame for later use'''
    manager.dump_frame(np.dstack(As), i)
    if i % 10 ==0:
      gc.collect()

    ''' show image in RGB '''
    img.set_array(np.dstack(As))
    return img,


  fig, ax = plt.subplots()
  img = ax.imshow(A, cmap='viridis', interpolation='nearest')

  ani = matplotlib.animation.FuncAnimation(
      fig, update, frames=5, interval=20, blit=True, repeat=False,
  )
  
  plt.pause(0.5)
  plt.close('all')
  gc.collect()
  
  
  
