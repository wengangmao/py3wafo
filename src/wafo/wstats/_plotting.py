"""Headless-capable statistical graphics and factorial-model diagnostics.

Plotting functions return Matplotlib axes or documented numerical results and
never call show(). Interactive point identification uses nonblocking callbacks.
"""
import numpy as np
from scipy import stats


def _axes(ax=None,projection=None):
    import matplotlib.pyplot as plt
    return plt.figure().add_subplot(111,projection=projection) if ax is None else ax


def _data(x):
    a=np.asarray(x,dtype=float)
    if a.size<2 or not np.all(np.isfinite(a)):
        raise ValueError('At least two finite observations are required')
    return a


def wqqplot(x,y,ps='+',*,ax=None):
    from . import wquantile
    x,y=np.sort(_data(x).ravel()),np.sort(_data(y).ravel())
    n=min(len(x),len(y)); p=(np.arange(n)+.5)/n
    ax=_axes(ax)
    ax.plot(wquantile(x,p),wquantile(y,p),ps)
    ax.set(xlabel='Data X',ylabel='Data Y',title='Quantile comparison')
    return ax


def distplot(x,dist,*,ax=None):
    from . import __dict__ as functions
    values=_data(x).ravel()
    stem=str(dist).lower().removesuffix('pdf')
    if stem+'fit' not in functions or stem+'inv' not in functions:
        raise ValueError('Distribution must have a WAFO fit and inverse function')
    parameters=np.atleast_1d(functions[stem+'fit'](values))
    theoretical=functions[stem+'inv']((np.arange(len(values))+.5)/len(values),*parameters)
    ax=wqqplot(values,theoretical,ax=ax)
    ax.set(title=stem+' probability plot',ylabel='Fitted distribution quantiles')
    return ax


def wnormplot(x,nr1=1,nr2=1,df=np.inf,*,ax=None):
    """Normal/Student probability paper; return axes and preserve columns."""
    a=_data(x); a=a[:,None] if a.ndim==1 else a
    if a.ndim!=2 or np.isnan(df) or df<=0:
        raise ValueError('Expected columns of observations and positive df')
    ax=_axes(ax); p=(np.arange(len(a))+.5)/len(a)
    score=stats.norm.ppf(p) if np.isinf(df) else stats.t.ppf(p,df)
    for column in a.T:
        column=np.sort(column); ax.plot(column,score,'.')
        if nr1:
            sd=column.std(ddof=1)
            if sd<=0:raise ValueError('Probability plot requires nonconstant data')
            limits=column[[0,-1]]; ax.plot(limits,(limits-column.mean())/sd,'--')
    if nr2:
        for q in (.01,.1,.5,.9,.99):
            ax.axhline(stats.norm.ppf(q) if np.isinf(df) else stats.t.ppf(q,df),color='.85',lw=.5)
    ax.set(xlabel='Data',ylabel='Standard normal quantiles' if np.isinf(df) else 'Student quantiles',title='Probability plot')
    return ax


def _paper(x,family,ax,full_output):
    from . import __dict__ as functions
    values=np.sort(_data(x).ravel()); p=(np.arange(len(values))+.5)/len(values)
    parameters=np.atleast_1d(functions['w'+family+'fit'](values)); ax=_axes(ax)
    # Quantile-versus-data plotting avoids unstable log-of-log display axes
    # while retaining the straight-line assessment on fitted probability paper.
    theoretical=functions['w'+family+'inv'](p,*parameters)
    ax.plot(values,theoretical,'.'); ends=values[[0,-1]]; ax.plot(ends,ends,'--')
    ax.set(xlabel='Data',ylabel='Fitted '+family+' quantile',title=family+' probability plot')
    result=parameters[0] if len(parameters)==1 else parameters
    return (result,ax) if full_output else result


def wexpplot(x,*,ax=None,full_output=False):
    return _paper(x,'exp',ax,full_output)


def wraylplot(x,*,ax=None,full_output=False):
    return _paper(x,'rayl',ax,full_output)


def wweibplot(x,*,ax=None,full_output=False):
    return _paper(x,'weib',ax,full_output)


def wgumbplot(x,*,ax=None,full_output=False):
    return _paper(x,'gumb',ax,full_output)


def whisto(x,N=None,odd=0,scale=0,lintype='b',*,ax=None,full_output=False):
    """Histogram with optional unit-area scaling; return uniform bin width."""
    a=_data(x).ravel(); N=int(np.ceil(np.sqrt(len(a)))) if N is None else N
    if int(N)!=N or N<1 or odd not in (0,1) or scale not in (0,1):
        raise ValueError('Expected positive integer bins and odd/scale in 0,1')
    width=np.ptp(a)/N
    if width==0:width=max(1,abs(a[0]))/N
    # Shift bins half a width when odd=1; always include all measurements.
    start=np.floor(a.min()/width-.5*odd)*width+.5*odd*width
    edges=start+np.arange(int(np.ceil((a.max()-start)/width))+2)*width
    ax=_axes(ax); counts,edges,_=ax.hist(a,bins=edges,density=bool(scale),histtype='step',color=lintype[0])
    ax.set(xlabel='Value',ylabel='Density' if scale else 'Count',title='Histogram')
    return (width,counts,edges,ax) if full_output else width


def pairs(x,labels=None,*,axes=None):
    import matplotlib.pyplot as plt
    a=_data(x)
    if a.ndim!=2:raise ValueError('Expected an observation-by-variable matrix')
    d=a.shape[1]
    axes=plt.subplots(d,d,squeeze=False)[1] if axes is None else np.asarray(axes)
    if axes.shape!=(d,d):raise ValueError('Axes must form a D-by-D array')
    labels=['x'+str(i+1) for i in range(d)] if labels is None else list(labels)
    if len(labels)!=d:raise ValueError('Provide one label per variable')
    for i in range(d):
        for j in range(d):
            if i==j:axes[i,j].hist(a[:,i],bins='auto',histtype='step')
            else:axes[i,j].plot(a[:,j],a[:,i],'.',ms=2)
            if i==d-1:axes[i,j].set_xlabel(labels[j])
            if j==0:axes[i,j].set_ylabel(labels[i])
    return axes


def kdeplot(data,g=None,plotflag=1,h=None,*,ax=None):
    from ..kdetools import kde,hldpi
    a=_data(data).ravel(); h=float(hldpi(a)[0]) if h is None else h
    x=np.linspace(a.min()-3*h,a.max()+3*h,257); f=kde(a,x,hs=h)
    if plotflag not in (0,1):raise ValueError('plotflag must be 0 or 1')
    if plotflag:
        ax=_axes(ax); ax.plot(x,f)
        if g is not None:
            g=np.asarray(g); ax.plot(g[:,0],g[:,1],'--')
        ax.set(xlabel='Value',ylabel='Density',title='Kernel density')
    return np.column_stack((x,f))


def fitmodel(y,model):
    """Fit selected factorial effects plus intercept in standard run order.

    Returns (fitted,residual,sample_residual_std,dof), matching the source's
    residual std normalization; dof=n-selected_terms-1 is separate.
    """
    from ._design import ffd,cnr2cl
    a=_data(y); n=a.shape[0]; k=int(np.log2(n))
    if 2**k!=n:raise ValueError('Run count must be a power of two')
    if np.asarray(model).size==0:
        labels=[]
    else:
        if np.asarray(model).dtype.kind not in 'US':model=cnr2cl(model)
        labels=[model] if isinstance(model,str) else list(model)
    labels=[''.join(sorted(str(label).strip().upper())) for label in labels]
    if len(set(labels))!=len(labels):raise ValueError('Duplicate model terms')
    design=2*ffd(k)-1; columns=[np.ones(n)]
    for label in labels:
        indices=[ord(letter)-65 for letter in label]
        if not indices or len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=k:
            raise ValueError('Model labels must identify distinct existing factors')
        columns.append(np.prod(design[:,indices],axis=1))
    X=np.column_stack(columns); coefficients=np.linalg.lstsq(X,a,rcond=None)[0]
    fitted=X@coefficients; residual=a-fitted
    return fitted,residual,np.std(residual,axis=0,ddof=1),n-X.shape[1]


def getmodel(ef,id,treshLow,treshUp=None):
    """Return selected factor labels and zero-based effect-label indices."""
    e=np.asarray(ef,dtype=float).ravel(); labels=np.asarray(id)
    upper=-treshLow if treshUp is None else treshUp
    if len(labels)!=len(e)-1 or treshLow>upper:raise ValueError('Effect labels or thresholds do not agree')
    selected=np.flatnonzero((e[1:]<treshLow)|(e[1:]>upper))
    return labels[selected],selected


def nplot(ef,id=None,offset=None,*,ax=None):
    from ._design import yates
    values=_data(ef).ravel()
    if id is None:_,id=yates(np.zeros_like(values),return_labels=True)
    labels=np.asarray(id)
    if len(labels)!=len(values)-1:raise ValueError('Provide one label per non-mean effect')
    order=np.argsort(values[1:]); effects=values[1:][order]
    score=stats.norm.ppf((np.arange(len(effects))+.5)/len(effects))
    ax=_axes(ax); ax.plot(effects,score,'.')
    shift=np.ptp(effects)/max(1,3*len(effects)) if offset is None else offset
    for effect,z,label in zip(effects,score,labels[order]):ax.text(effect+shift,z,str(label))
    ax.set(xlabel='Factorial effect',ylabel='Normal score',title='Effect probability plot')
    return ax


def cplot(y,D=None,*,ax=None):
    """Response cube for two or three factors (other sizes are explicit errors)."""
    from ._design import ffd
    values=_data(y).ravel(); k=int(np.log2(len(values)))
    design=ffd(k) if D is None else np.asarray(D,dtype=float)
    if design.shape[0]!=len(values) or design.ndim!=2 or design.shape[1] not in (2,3):
        raise ValueError('Cube plots require two or three design factors')
    d=design.shape[1]; design=np.where(design>=1,1.,-1.)
    ax=_axes(ax,'3d' if d==3 else None)
    for i in range(len(design)):
        for j in range(i):
            if np.sum(design[i]!=design[j])==1:
                segment=np.vstack((design[j],design[i])); ax.plot(*segment.T,color='.5')
        ax.text(*design[i],str(values[i]))
    ax.set(xlabel='A',ylabel='B',title='Response cube')
    if d==3:ax.set_zlabel('C')
    return ax


def wgumbtest(x,a=.05,*,full_output=False):
    """PWM GEV-shape test of Gumbel, with a corrected two-sided p-value."""
    values=np.sort(_data(x).ravel()); n=len(values)
    if n<3 or np.ptp(values)==0 or not 0<a<1:raise ValueError('Need at least 3 nonconstant observations and 0<a<1')
    ranks=np.arange(n); b0=values.mean()
    b1=np.mean(ranks/(n-1)*values); b2=np.mean(ranks*(ranks-1)/((n-1)*(n-2))*values)
    if 3*b2-b0<=0:raise ValueError('Degenerate probability-weighted moments')
    z=(2*b1-b0)/(3*b2-b0)-np.log(2)/np.log(3)
    shape=7.859*z+2.9554*z*z; statistic=shape*np.sqrt(n/.5633)
    p=2*stats.norm.sf(abs(statistic))
    return dict(pvalue=p,shape=shape,statistic=statistic,reject=bool(p<a),alpha=a) if full_output else p


class PointIdentifier:
    """Nonblocking plot picker with selected zero-based indices."""
    def __init__(self,x,y,ps,labels,ax):
        self.ax=_axes(ax); self.x,self.y=np.broadcast_arrays(x,y)
        self.x,self.y=self.x.ravel(),self.y.ravel()
        if not len(self.x) or not np.all(np.isfinite([self.x,self.y])):raise ValueError('Expected finite points')
        self.labels=[str(i) for i in range(len(self.x))] if labels is None else list(labels)
        if len(self.labels)!=len(self.x):raise ValueError('Provide one label per point')
        self.artist=self.ax.plot(self.x,self.y,ps,picker=5)[0]; self.indices=[]
        self.connection=self.ax.figure.canvas.mpl_connect('pick_event',self._pick)
    def _pick(self,event):
        if event.artist is self.artist and len(event.ind):self.select(int(event.ind[0]))
    def select(self,index):
        if not 0<=index<len(self.x):raise IndexError(index)
        self.indices.append(index); self.ax.annotate(str(self.labels[index]),(self.x[index],self.y[index])); return index
    def disconnect(self):
        self.ax.figure.canvas.mpl_disconnect(self.connection)


def identify(x,y,ps='*',text=None,*,ax=None):
    """Attach a nonblocking clickable picker; returns a PointIdentifier."""
    return PointIdentifier(x,y,ps,text,ax)


def _groups(V,H,condon,res):
    v,h=_data(V).ravel(),_data(H).ravel()
    if v.shape!=h.shape or condon not in (1,2):raise ValueError('Expected paired data and condon=1/2')
    condition,response=(h,v) if condon==2 else (v,h)
    width=np.ptp(condition)/9 if res is None else res
    if not np.isscalar(width) or not np.isfinite(width) or width<=0:raise ValueError('Positive conditioning-bin width required')
    origin=np.floor(condition.min()/width)*width
    group=np.floor((condition-origin)/width).astype(int)
    result=[]
    for index in np.unique(group):
        data=np.sort(response[group==index])
        if len(data)>=5:result.append((origin+(index+.5)*width,data))
    if not result:raise ValueError('No conditioning bin has five observations; increase res')
    return result


def _conditional_cdf_plot(family,V,H,phat,condon,res,plotflag,figdata,sym,axes):
    from . import __dict__ as functions
    import matplotlib.pyplot as plt
    groups=_groups(V,H,condon,res)
    if plotflag not in (0,1,2,3):raise ValueError('Conditional CDF plot modes are 0 through 3')
    styles=('b.','r--') if sym is None else sym
    if len(styles)!=2:raise ValueError('Provide empirical and theoretical styles')
    curves=[]
    for center,data in groups:
        empirical=(np.arange(len(data))+.5)/len(data)
        if phat is None:model=None
        elif condon==2:model=functions[family+'cdf'](data,np.full_like(data,center),phat,condon=2)
        else:model=functions[family+'cdf'](np.full_like(data,center),data,phat,condon=1)
        curves.append(dict(condition=center,x=data,empirical=empirical,model=model))
    if not plotflag:return curves
    if axes is None:
        layout=np.asarray(figdata,dtype=int)
        if layout.shape!=(3,) or np.any(layout<1):raise ValueError('figdata must be [rows,cols,figures]')
        rows,cols,figures=layout
        if len(curves)>rows*cols*figures:raise ValueError('Not enough panels; increase figdata or res')
        created=[]
        for _ in range(figures):created.extend(plt.subplots(rows,cols,squeeze=False)[1].ravel())
        axes=created
    axes=np.asarray(axes,dtype=object).ravel()
    if len(axes)<len(curves):raise ValueError('Provide enough axes for all conditioning groups')
    for ax,curve in zip(axes,curves):
        data=curve['x']
        for values,style in zip((curve['empirical'],curve['model']),styles):
            if values is None:continue
            if plotflag==1:ax.plot(data,values,style)
            elif plotflag==2:ax.semilogy(data,1-values,style)
            else:
                if np.any(data<=0):raise ValueError('Log-log conditional plots require positive responses')
                ax.loglog(data,1-values,style)
        ax.set(title=f'X{condon} near {curve["condition"]:.3g}',xlabel='Response',ylabel='CDF' if plotflag==1 else 'Exceedance probability')
    for ax in axes[len(curves):]:ax.set_visible(False)
    return list(axes[:len(curves)])


def dist2dcdfplot(V,H,phat=None,condon=2,res=None,plotflag=1,figdata=(3,3,1),sym=None,*,axes=None):
    """Binned empirical/conditional-model CDF panels; mode 0 returns curves."""
    return _conditional_cdf_plot('dist2d',V,H,phat,condon,res,plotflag,figdata,sym,axes)


def mdist2dcdfplot(V,H,phat=None,condon=2,res=None,plotflag=1,figdata=(3,3,1),sym=None,*,axes=None):
    return _conditional_cdf_plot('mdist2d',V,H,phat,condon,res,plotflag,figdata,sym,axes)


def weib2dcdfplot(V,H,phat=None,condon=2,res=None,plotflag=1,figdata=(3,3,1),sym=None,*,axes=None):
    return _conditional_cdf_plot('weib2d',V,H,phat,condon,res,plotflag,figdata,sym,axes)


def _conditional_stat_plot(family,V,H,phat,condon,res,ax,full_output):
    from . import __dict__ as functions
    groups=_groups(V,H,condon,res)
    empirical=np.asarray([[center,data.mean(),data.std(ddof=1),len(data)] for center,data in groups])
    ax=_axes(ax); ax.plot(empirical[:,0],empirical[:,1],'x',label='sample mean'); ax.plot(empirical[:,0],empirical[:,2],'o',label='sample std')
    model=None
    if phat is not None:
        mean,variance=functions[family+'stat'](phat,condon=condon,cvar=empirical[:,0])
        model=np.column_stack((empirical[:,0],mean,np.sqrt(variance)))
        ax.plot(model[:,0],model[:,1],'-',label='model mean'); ax.plot(model[:,0],model[:,2],'--',label='model std')
    ax.set(xlabel='Conditioning variable',ylabel='Conditional moment',title='Conditional mean and standard deviation'); ax.legend()
    return (ax,empirical,model) if full_output else ax


def dist2dstatplot(V,H,phat=None,condon=2,res=None,*,ax=None,full_output=False):
    return _conditional_stat_plot('dist2d',V,H,phat,condon,res,ax,full_output)


def mdist2dstatplot(V,H,phat=None,condon=2,res=None,*,ax=None,full_output=False):
    return _conditional_stat_plot('mdist2d',V,H,phat,condon,res,ax,full_output)


def weib2dstatplot(V,H,phat=None,condon=2,res=None,*,ax=None,full_output=False):
    return _conditional_stat_plot('weib2d',V,H,phat,condon,res,ax,full_output)


def dist2dparamplot(PV1,sPV1=None,plotflag=4,*,axes=None):
    """Plot fitted/smoothed conditional parameter columns; mode 0 returns table."""
    import matplotlib.pyplot as plt
    table=np.asarray(PV1['x'][0],dtype=float)
    if table.ndim!=2 or table.shape[1]<2:raise ValueError('Expected conditioning/parameter table in x[0]')
    if plotflag==0:return table.copy()
    if plotflag not in (1,2,3,4):raise ValueError('Parameter plot modes are 0 through 4')
    columns=list(range(1,table.shape[1])) if plotflag==4 else [plotflag]
    if max(columns)>=table.shape[1]:raise ValueError('Selected parameter is absent')
    axes=plt.subplots(len(columns),1,squeeze=False)[1].ravel() if axes is None else np.asarray(axes,dtype=object).ravel()
    if len(axes)!=len(columns):raise ValueError('Provide one axes per selected parameter')
    for ax,column in zip(axes,columns):
        ax.plot(table[:,0],table[:,column],'o',label='fitted')
        if sPV1 is not None:
            smoothed=np.asarray(sPV1['x'][0],dtype=float)
            if smoothed.shape[1]!=table.shape[1]:raise ValueError('Smoothed parameter columns do not match')
            ax.plot(smoothed[:,0],smoothed[:,column],'-',label='smoothed')
        ax.set(xlabel='Conditioning variable',ylabel='Parameter '+str(column)); ax.legend()
    return list(axes)
