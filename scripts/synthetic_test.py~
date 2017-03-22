''' 
This script generates synthetic data consisting of white and random
walk noise, and then the noise is characterized using the ML method
and the REML method.
'''
import numpy as np
from rbf.gauss import GaussianProcess,gpse,gpexp,gppoly
import matplotlib.pyplot as plt
from rbf.gauss import _trisolve,_cholesky,_assert_shape
from scipy.signal import periodogram
from scipy.optimize import fmin
import time as timemod
from pygeons.mp import parmap
np.random.seed(0)

def fmin_pos(func,x0,*args,**kwargs):
  '''fmin with positivity constraint'''
  def pos_func(x,*args):
    return func(np.exp(x),*args)

  out = fmin(pos_func,np.log(x0),*args,**kwargs)
  out = np.exp(out)
  return out

def restricted_likelihood(d,mu,sigma,p):
  ''' 
  Parameters
  ----------
  d : (N,) array
    observations.
  
  mu : (N,) array
    mean of the random vector.
  
  sigma : (N,N) array    
    covariance of the random vector.
  
  p : (N,P) array, optional  
    Improper basis vectors.

  '''
  d = np.asarray(d,dtype=float)
  _assert_shape(d,(None,),'d')

  mu = np.asarray(mu,dtype=float)
  _assert_shape(mu,(d.shape[0],),'mu')

  sigma = np.asarray(sigma,dtype=float) # data covariance
  _assert_shape(sigma,(d.shape[0],d.shape[0]),'sigma')

  p = np.asarray(p,dtype=float)
  _assert_shape(p,(d.shape[0],None),'p')

  n,m = p.shape
  A = _cholesky(sigma,lower=True)      
  B = _trisolve(A,p,lower=True)        
  C = _cholesky(B.T.dot(B),lower=True) 
  D = _cholesky(p.T.dot(p),lower=True) 
  a = _trisolve(A,d-mu,lower=True)     
  b = _trisolve(C,B.T.dot(a),lower=True) 
  out = (np.sum(np.log(np.diag(D))) -
         np.sum(np.log(np.diag(A))) -
         np.sum(np.log(np.diag(C))) -
         0.5*a.T.dot(a) +
         0.5*b.T.dot(b) -
         0.5*(n-m)*np.log(2*np.pi))
  
  return out

def likelihood(d,mu,sigma,p):
  ''' 
  Parameters
  ----------
  d : (N,) array
    observations.
  
  mu : (N,) array
    mean of the random vector.
  
  sigma : (N,N) array    
    covariance of the random vector.
  
  p : (N,P) array, optional  
    Improper basis vectors.

  '''
  d = np.asarray(d,dtype=float)
  _assert_shape(d,(None,),'d')

  mu = np.asarray(mu,dtype=float)
  _assert_shape(mu,(d.shape[0],),'mu')

  sigma = np.asarray(sigma,dtype=float) # data covariance
  _assert_shape(sigma,(d.shape[0],d.shape[0]),'sigma')

  p = np.asarray(p,dtype=float)
  _assert_shape(p,(d.shape[0],None),'p')

  n,m = p.shape
  A = _cholesky(sigma,lower=True)      
  B = _trisolve(A,p,lower=True)        
  C = _cholesky(B.T.dot(B),lower=True) 
  a = _trisolve(A,d-mu,lower=True)     
  b = _trisolve(C,B.T.dot(a),lower=True) 
  out = (-np.sum(np.log(np.diag(A))) -
          0.5*a.T.dot(a) +
          0.5*b.T.dot(b) -
          0.5*n*np.log(2*np.pi))

  return out

def gpbrown(coeff):
  def mean(x):
    return np.zeros(x.shape[0])

  def cov(x1,x2):
    x1,x2 = np.meshgrid(x1,x2,indexing='ij')
    out = coeff*np.min([x1,x2],axis=0)
    return out 

  out = GaussianProcess(mean,cov,dim=1)
  return out
                    
# sampling period in years
rw_scale = 1.3
w_scale = 1.1
dt = 1.0/365.25
time = np.arange(dt,2.5,dt)

noise_true  = gpbrown(rw_scale**2)
noise_true += gpexp((0.0,w_scale**2,1e-10)) 

# view a sample in the time and frequency domain to make sure
# everything looks ok
mu,sigma = noise_true(time[:,None]) 
sample = noise_true.draw_sample(time[:,None])
freq,pow = periodogram(sample,1.0/dt)
freq,pow = freq[1:],pow[1:]
pow_true = rw_scale**2/(2*np.pi**2*freq**2) + 2*w_scale**2*dt

fig,axs = plt.subplots(2,1)
axs[0].plot(time,mu,color='k',zorder=2,label='expected')
axs[0].fill_between(time,mu-sigma,mu+sigma,color='k',alpha=0.2,zorder=0,label='std. dev.')
axs[0].grid(ls=':',c='0.5')
axs[0].plot(time,sample,color='C0',zorder=1,lw=1.0,label='sample')
axs[0].set_xlim((time.min(),time.max()))
axs[0].set_xlabel('time [yr]')
axs[0].set_ylabel('displacement [mm]')
axs[0].legend()

axs[1].set_xlim((freq.min(),freq.max()))
axs[1].loglog(freq,pow,color='C0',label='sample',lw=1.0,zorder=1)
axs[1].loglog(freq,pow_true,color='k',label='expected',zorder=2)
axs[1].grid(ls=':',c='0.5')
axs[1].set_xlabel('frequency [1/yr]')
axs[1].set_ylabel('power density [$\mathregular{mm^2 \cdot yr}$]')
axs[1].legend()
fig.tight_layout()
plt.show()

# generate data sets
data_sets = [noise_true.draw_sample(time[:,None]) for i in range(100)]
ml_file = open('ml_estimates.txt','w')
ml_file.write('timeseries-length mean %s\n' % ' '.join(np.arange(0,105,5).astype(str)))
ml_file.flush()
reml_file = open('reml_estimates.txt','w')
reml_file.write('timeseries-length mean %s\n' % ' '.join(np.arange(0,105,5).astype(str)))
reml_file.flush()

def estimate_scales(tslength):
  print('estimating random walk scale for timeseries length %s' % tslength)
  def ml_objective(theta,d,cov_rw,cov_w,p):
    cov = theta**2*cov_rw + w_scale**2*cov_w
    mu = np.zeros(d.shape[0])
    return -likelihood(d,mu,cov,p)

  def reml_objective(theta,d,cov_rw,cov_w,p):
    cov = theta**2*cov_rw + w_scale**2*cov_w
    mu = np.zeros(d.shape[0])
    return -restricted_likelihood(d,mu,cov,p)

  # time indices to use
  ml_solns = []
  reml_solns = []
  idx = time < tslength

  P      = gppoly(1).basis(time[idx,None])
  COV_RW = gpbrown(1.0).covariance(time[idx,None],time[idx,None])
  COV_W  = np.eye(sum(idx))
  for data in data_sets:
    ans = fmin_pos(ml_objective,[1.0],args=(data[idx],COV_RW,COV_W,P),disp=False) 
    ml_solns += [ans[0]]
    ans = fmin_pos(reml_objective,[1.0],args=(data[idx],COV_RW,COV_W,P),disp=False) 
    reml_solns += [ans[0]]
  
  # compute statistics on solution
  mean = np.mean(ml_solns)
  percs = np.percentile(ml_solns,np.arange(0,105,5))
  entry = '%s %s %s\n' % (tslength,mean,' '.join(percs.astype(str)))
  ml_file.write(entry)
  ml_file.flush()

  mean = np.mean(reml_solns)
  percs = np.percentile(reml_solns,np.arange(0,105,5))
  entry = '%s %s %s\n' % (tslength,mean,' '.join(percs.astype(str)))
  reml_file.write(entry)
  reml_file.flush()

tslengths = np.arange(0.1,2.6,0.1)
parmap(estimate_scales,tslengths)
ml_file.close()
reml_file.close()
    
