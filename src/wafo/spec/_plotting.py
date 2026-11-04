"""Matplotlib plotting for spectral structures (optional plot dependency)."""
import numpy as np
from scipy.integrate import trapezoid
from ._transforms import _frequency,spec2spec


def wspecplot(S,plotflag=1,linetype='-',*,ax=None):
    """Plot 1D, directional, or Cartesian-wavenumber spectral densities.

    Implements the documented numerical plot modes 1–3 (1D), 1–7 (directional)
    and 1–2 (Cartesian). Returns a Matplotlib Axes or list of Axes. Automatic
    MATLAB point-and-click text editing is not reproduced. No show() is called.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('wspecplot requires the optional matplotlib dependency') from exc
    if isinstance(S,(list,tuple)):
        if not S:
            raise ValueError('No spectra to plot')
        handles=[]
        for spectrum in S:
            current=wspecplot(spectrum,plotflag,linetype,ax=ax)
            handles.append(current)
            ax=current
        return handles[-1]
    if plotflag!=int(plotflag):
        raise ValueError('plotflag must be an integer')
    plotflag=int(plotflag)
    kind=str(S.get('type','freq')).lower()
    if kind=='k2d':
        k1,k2,density=np.asarray(S['k']),np.asarray(S['k2']),np.asarray(S['S'])
        if density.shape!=(len(k2),len(k1)) or not np.all(np.isfinite(density)):
            raise ValueError('Invalid Cartesian spectral grid')
        if plotflag not in (1,2):
            raise ValueError('Cartesian plot modes are 1 and 2')
        ax=plt.subplots()[1] if ax is None else ax
        (ax.contour if plotflag==1 else ax.contourf)(k1,k2,density)
        ax.set(xlabel='k1 (rad/m)',ylabel='k2 (rad/m)',title=str(S.get('note','')))
        return ax
    key,x,density=_frequency(S)
    if density.ndim==1:
        if plotflag not in (1,2,3):
            raise ValueError('One-dimensional plot modes are 1, 2, and 3')
        axes=plt.subplots(2,1)[1] if plotflag==3 and ax is None else ax
        if plotflag!=3:
            axes=plt.subplots()[1] if ax is None else ax
            axes=[axes]
        elif axes is None or not hasattr(axes,'__len__') or len(axes)!=2:
            raise ValueError('Mode 3 requires two supplied axes or no ax argument')
        for i,axis in enumerate(axes):
            db=plotflag==2 or (plotflag==3 and i==1)
            with np.errstate(divide='ignore'):
                values=10*np.log10(density) if db else density
            axis.plot(x,values,linetype)
            axis.set(xlabel={'w':'angular frequency (rad/s)','f':'frequency (Hz)','k':'wavenumber (rad/m)'}[key],
                     ylabel='10 log10 S' if db else 'spectral density',title=str(S.get('note','')))
        return list(axes) if plotflag==3 else axes[0]
    if kind not in ('dir','encdir') or density.ndim!=2:
        raise ValueError('Expected a 1D, directional, or Cartesian spectrum')
    if plotflag not in range(1,8):
        raise ValueError('Directional plot modes are integers from 1 through 7')
    theta=np.asarray(S['theta'],dtype=float)
    if density.shape!=(len(theta),len(x)):
        raise ValueError('Directional grid and density do not match')
    phi=S.get('phi',0)
    if S.get('theta_units','radians')=='degrees' or np.ptp(theta)>3*np.pi:
        theta=theta*np.pi/180
        phi=phi*np.pi/180
        density=density*180/np.pi
    theta=theta+phi
    if plotflag in (2,3):
        if ax is None:
            figure=plt.figure()
            axes=[figure.add_subplot(121),figure.add_subplot(122,projection='polar')]
        elif not hasattr(ax,'__len__') or len(ax)!=2:
            raise ValueError('Directional modes 2/3 require two supplied axes or no ax argument')
        else:
            axes=ax
        marginal=trapezoid(density,x=theta,axis=0)
        axes[0].plot(x,marginal,linetype)
        values=density if plotflag==2 else np.divide(density,marginal[None,:],out=np.zeros_like(density),where=marginal[None,:]>0)
        direction=trapezoid(values,x=x,axis=1)
        axes[1].plot(theta,direction,linetype)
        axes[0].set(xlabel=key,ylabel='spectral density')
        axes[1].set_title('directional energy' if plotflag==2 else 'integrated directional spreading')
        return list(axes)
    if ax is None:
        ax=plt.figure().add_subplot(111,projection='polar' if plotflag==1 else '3d' if plotflag in (4,5) else None)
    if plotflag==1:
        ax.contourf(theta,x,density.T)
    elif plotflag in (4,5):
        frequency,angle=np.meshgrid(x,theta)
        xx,yy=(frequency,angle) if plotflag==4 else (frequency*np.cos(angle),frequency*np.sin(angle))
        ax.plot_surface(xx,yy,density)
    elif plotflag==6:
        ax.contour(x,theta,density)
    else:
        ax.contourf(x,theta,density)
    ax.set_title(str(S.get('note','')))
    return ax
