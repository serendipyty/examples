# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:11:52 2016

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

# %%
%load_ext autoreload
%autoreload 2

#%aimport pymh
# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from timeit import default_timer as timer

import scipy.io

from serendipyty.seismic.modelling import awe2d
#import ebc
#import modeling.util
from serendipyty.seismic.utils.util import ricker
from serendipyty.seismic.utils.fd import stability
from serendipyty.seismic.utils.fd import dispersion
from serendipyty.seismic.utils.util import rectangle
from serendipyty.seismic.utils.util import oneface

from serendipyty.seismic.utils.dispersion import itdt3

from serendipyty.seismic.input.wavelets import RickerWavelet

#from pymh import *

from joblib import Parallel, delayed
import multiprocessing

import sys

sys.dont_write_bytecode = True

#import __builtin__
#from IPython.lib import deepreload
#__builtin__.reload = deepreload.reload


DTYPE = np.float64

#if __name__ == '__main__':

import matplotlib as mpl
#mpl.rc('image', interpolation='none', origin='lower', cmap = 'gray')
mpl.rc('image', interpolation='none')
plt.rcParams['figure.figsize'] = 10, 8
#plt.rcParams['figure.figsize'] = 45, 30
plt.rcParams['font.size'] = 14

# %% Parameters

# Extent in # of cells
nx = 201
nz = 101

# Sampling in m
dx = DTYPE(5)
dz = DTYPE(5)

# Source in grid point locations
src_loc = np.array([600/dx, 0, 250/dx, 1], dtype=np.uint)

# Source
sourcetype = 'q'
# Central frequency
fc = np.float64(30)

# PML
pml = 4
npml = 30

# %% Model building

vp0 = 2000
rho0 = 1000

vp = np.ones((nx, nz), dtype=DTYPE)*vp0
rho = np.ones((nx, nz), dtype=DTYPE)*rho0

x = np.arange(nx)*dx
z = np.arange(nz)*dz

dip = 0.1

inta = (-dip)*x + 150.0

mask = np.zeros((nx,nz), dtype=np.bool)
for i in range(nx):
    mask[i, np.rint(inta[i]/dx).astype(int):] = True

vp[mask] = 2500
rho[mask] = 1500

intb = 350
vp[:,np.rint(intb/dx).astype(int):] = 2300
rho[:,np.rint(intb/dx).astype(int):] = 1300

fig = plt.figure(facecolor='w', edgecolor='k', figsize=(4,5))
plt.imshow(rho.T) #, extent=[0, x[-1], z[-1], 0])
plt.colorbar()

#plt.show()

# %% Dispersion and Stability

dx_no_dispersion = dispersion(vp.min(), dx, fc, coeff=2.0)
print('dx_no_dispresion is ', dx_no_dispersion)

# Time
tmax = 0.42
dt = np.float64(0.001)

# I compute the stability criterion
dt_stable = stability(vp.max(), dx, dt)
print('dt_stable is ', dt_stable)

# For this example, I keep dt (and nt) fixed
#dt = dt_stable
nt = int(tmax/dt)+1

t = np.arange(nt)*dt

# %% Wavelets

# Ricker
q_src = ricker(fc, nt, dt)

# Delta
delta_src = np.zeros((nt,), dtype=DTYPE)
delta_src[0] = 1.0

plt.plot(t, q_src)
#plt.show()

wav2 = RickerWavelet(t, fc)

q_src2 = PointSource(src_loc, wav2, 'mono')

# %% Receivers

# Number of disjoint subdomains
nsub = 1
semt_origins = np.ndarray((nsub, 4), dtype=np.uint)
srec_origins = np.ndarray((nsub, 4), dtype=np.uint)

# order: xemt1 xemt2 zemt1 zemt2
#semt_origins[0,:] = (40,  80,  40,  60)
#semt_origins[1,:] = (120, 160,  10,  30)
# order: xemt zemt nxemt nzemt
semt_origins[0,:] = (0,  35,  nx,  nz-35)

# number of gridpoints between recording and emitting surface
ngpts = 2

srec_origins = semt_origins + np.array((0, ngpts, 0, 0), dtype=np.uint)

# Locations
locations = []
for i in range(nsub):
    locations = locations + oneface(faces=(2,), origin=(semt_origins[i][0], 0, semt_origins[i][1]),
          number_of_cells=(semt_origins[i][2], 0, semt_origins[i][3]),
          cell_size=(1,1,1)
          )
semt_locs = np.array(locations, dtype=np.uint)

locations = []
for i in range(nsub):
    locations = locations + oneface(faces=(2,), origin=(srec_origins[i][0], 0, srec_origins[i][1]),
          number_of_cells=(srec_origins[i][2], 0, srec_origins[i][3]),
          cell_size=(1,1,1)
          )
srec_locs = np.array(locations, dtype=np.uint)

nemt = semt_locs.shape[0]
nrec = srec_locs.shape[0]

# %% Configuration

fig = plt.figure(facecolor='w', edgecolor='k')
gs = gridspec.GridSpec(1, 2)
gs.update(wspace=0.02)

ax0 = fig.add_subplot(gs[0, 0])
im0 = ax0.imshow(rho.T, cmap='seismic', interpolation='none', extent=[0, x[-1], z[-1], 0], aspect=1.5)
ax0.plot(semt_locs[::4,0]*dx, semt_locs[::4,2]*dx-dx, ls='None', marker='o')
ax0.plot(srec_locs[::4,0]*dx, srec_locs[::4,2]*dx+dx, ls='None', marker='p')
#ax0.plot(src_loc[0], src_loc[2], marker='p')
ax0.set_xlabel('Horizontal location [m]')
ax0.set_ylabel('Depth [m]')
ax0.set_xticks(ax0.get_xticks()[:4:])
ax0.set_yticks(ax0.get_yticks()[::2])

ax0.text(40, 350, r'$V$', bbox={'facecolor':'white', 'pad':10})
ax0.text(40, 100, r'$V^{\,\prime}$', bbox={'facecolor':'white', 'pad':10})

ax0.text(850, 137, r'$S^{\,emt}$', bbox={'facecolor':'white', 'pad':10})
ax0.text(850, 253, r'$S^{\,rec}$', bbox={'facecolor':'white', 'pad':10})

ax0.text(320, 100, r'$dx=1$m', bbox={'facecolor':'white', 'pad':10})
ax0.text(320, 350, r'$dx=5$m', bbox={'facecolor':'white', 'pad':10})

ax0.arrow(790, 175, 0, -50, head_width=20, head_length=10, fc='k', ec='k')

gs.tight_layout(fig)

#plt.savefig('configuration1.png', bbox_inches='tight', dpi=600, transparent=True)

# %%BCs for EBC algorithm
# free surface or rigid boundary or mixed BCs (bc = 'free'/'rigid'/'mixed') at the boundaries of the truncated domain
#    bc = 'free'
#bc = 'free'
bc = 'rigid'

# %% Run simulation

outparam = []
outparam.append({'type': 'slice',
                 'timestep_increment': 10
                })
outparam.append({'type': 'sub_volume_boundary',
                'attribute': 'p',
                'receiver_locations': semt_locs,
                'stagger_on_sub_volume': True,
                'timestep_increment': 1
                })
outparam.append({'type': 'sub_volume_boundary',
                'attribute': 'vn',
                'receiver_locations': semt_locs,
                'stagger_on_sub_volume': True,
                'timestep_increment': 1
                })
if bc == 'free':
    outparam.append({'type': 'sub_volume_boundary',
                    'attribute': 'p',
                    'receiver_locations': semt_locs,
                    'stagger_on_sub_volume': False,
                    'timestep_increment': 1
                    })

outputs = [None] * len(outparam)
outputs = awe2d.calc_full(vp,rho,dx,src_loc,q_src,fc,dt,'q',outparam,pml,npml,12)

# %% Plot snapshot

vmin, vmax = (-5e3, 5e3)

it = 20
snap = outputs['slice'][0, it, ...]

fig = plt.figure(facecolor='w', edgecolor='k')
gs = gridspec.GridSpec(1, 2)
gs.update(wspace=0.02)

ax0 = fig.add_subplot(gs[0, 0])
im0 = ax0.imshow(snap.T, cmap='seismic', vmin=vmin, vmax=vmax, interpolation='none', extent=[0, x[-1], z[-1], 0], aspect=1)
ax0.set_xlabel('Horizontal location [m]')
ax0.set_ylabel('Depth [m]')
ax0.set_xticks(ax0.get_xticks()[:4:])
ax0.set_yticks(ax0.get_yticks()[::2])
ax0.text(40, 460, 'e)')

ax0.annotate('artefacts', xy=(600, 220), xytext=(500, 430),
            arrowprops=dict(facecolor='black', shrink=0.05))

gs.tight_layout(fig)

#plt.savefig('snapshots1.png', bbox_inches='tight', dpi=600)
