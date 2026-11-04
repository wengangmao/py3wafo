"""Native-code capability ports: run after installing wafo-py3 0.2.0."""
import numpy as np
from wafo.native import native_files, resolve_native
from wafo.trgauss import rind, mvtprb, spec2thpdf
from wafo.spec import jonswap
from wafo.damage import lc2dplus

print('Native source/binary mappings:', len(native_files()))
rho = .3
value,error,_,info = rind([[1,rho],[rho,1]],[0,0],0,np.inf,[0,2],Nt=0,
    options={'seed':7,'maxpts':65536},full_output=True)
analytic = (rho*(np.pi/2+np.arcsin(rho))+np.sqrt(1-rho*rho))/(2*np.pi)
print('Positive derivative-product expectation:',value,'analytic:',analytic)
print('Estimated integration error:',error,'inform:',info['inform'])
print('Noncentral t rectangle:',mvtprb(np.eye(2),[-1,-1],[2,2],5,delta=[.5,1],seed=7))
oscillator = resolve_native('simduff.exe')(200,.001,.1,seed=123456789)
print('Direct Fortran-capability oscillator rows:',oscillator.shape)
wave = spec2thpdf(jonswap(sdata=[4,8]),definition='Tc',paramt=[0,12,7],
    options={'seed':7,'maxpts':2048})
print('Coarse half-wave period subdensity:',wave.f)
print('Triangle crossing reconstructed damage:',lc2dplus([[-1,0],[0,1],[1,0]],[1,2,3],1000))
