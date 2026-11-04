"""Hidden-regime time-series plotting."""
import numpy as np


def hmmplot(x,z,t=None,z_range=None,Title='',Ylabel='',colType=0,fontsize=None,*,axes=None):
    """Plot load and zero-based hidden regime; returns the two axes."""
    import matplotlib.pyplot as plt
    x,z=np.asarray(x,dtype=float),np.asarray(z,dtype=float)
    t=np.arange(len(x)) if t is None else np.asarray(t,dtype=float)
    if x.ndim!=1 or x.shape!=z.shape or x.shape!=t.shape or not len(x) or not np.all(np.isfinite([x,z,t])) or np.any(np.diff(t)<=0) or colType not in (0,1):
        raise ValueError('Expected matching finite load, regime and increasing time vectors')
    axes=plt.subplots(2,1,sharex=True)[1] if axes is None else axes
    if len(axes)!=2:raise ValueError('Provide two axes')
    if colType:
        for state in np.unique(z):
            axes[0].plot(t,np.where(z==state,x,np.nan),label=str(state))
    else:axes[0].plot(t,x)
    axes[1].step(t,z,where='post')
    axes[0].set(title=Title,ylabel=Ylabel); axes[1].set(xlabel='Time',ylabel='Regime')
    if z_range is not None:axes[1].set_ylim(z_range)
    if fontsize is not None:
        for axis in axes:axis.tick_params(labelsize=fontsize)
    return axes
