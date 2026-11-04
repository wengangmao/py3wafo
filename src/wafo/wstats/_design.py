"""Two-level factorial-design routines from WAFO's statistics module.

Factor labels retain their mathematical one-based numbering (A=1,...,z=52).
Arrays use ordinary Python indexing. Optional extra outputs are explicit.
"""
import itertools
import numpy as np

_LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def cl2cnr(labels):
    """Convert design labels such as '-ABC' to signed integer rows."""
    if not isinstance(labels,str) and np.asarray(labels).dtype.kind not in 'US':
        return np.asarray(labels,dtype=int)
    labels=[labels] if isinstance(labels,str) else list(labels)
    rows=[]
    for label in labels:
        sign=-1 if str(label).count('-')%2 else 1
        chars=[c for c in str(label) if c not in ' -+']
        if any(c not in _LETTERS for c in chars):
            raise ValueError('Labels contain an invalid factor character')
        row=sorted(_LETTERS.index(c)+1 for c in chars)
        if row:
            row[-1]*=sign
        rows.append(row)
    width=max(map(len,rows),default=0)
    return np.asarray([row+[0]*(width-len(row)) for row in rows],dtype=int).reshape(len(rows),width)


def cnr2cl(numbers):
    """Convert signed factor rows to a list of design-word labels."""
    a=np.atleast_2d(numbers)
    if not np.all(np.isfinite(a)) or np.any(a!=np.floor(a)) or np.any(abs(a)>52):
        raise ValueError('Factor numbers must be integers in [-52,52]')
    result=[]
    for row in a.astype(int):
        nz=row[row!=0]
        result.append(('-' if np.prod(np.sign(nz))<0 else '')+''.join(_LETTERS[i-1] for i in abs(nz)))
    return result


def _words(I):
    a=cl2cnr(I) if isinstance(I,str) or np.asarray(I).dtype.kind in 'US' else np.atleast_2d(I)
    if not np.all(np.isfinite(a)) or np.any(a!=np.floor(a)) or np.any(abs(a)>52):
        raise ValueError('Invalid generator numbers')
    words=[]
    for row in a.astype(int):
        word=0;sign=1
        for number in row:
            number=int(number)
            if number:
                word^=1<<(abs(number)-1)
                sign*=int(np.sign(number))
        if not word:
            raise ValueError('Generator words cannot be the identity')
        words.append((word,sign))
    return words


def _rows(words):
    rows=[]
    for mask,sign in words:
        mask=int(mask)
        row=[i+1 for i in range(mask.bit_length()) if mask>>i&1]
        if row:
            row[-1]*=sign
        rows.append(row)
    width=max(map(len,rows),default=0)
    return np.asarray([r+[0]*(width-len(r)) for r in rows],dtype=int).reshape(len(rows),width)


def cdr(I,*,return_resolution=False):
    """Complete defining relation from independent signed generator words."""
    group={0:1}
    for mask,sign in _words(I):
        extra={key^mask:value*sign for key,value in list(group.items())}
        if any(key in group and group[key]!=value for key,value in extra.items()):
            raise ValueError('Inconsistent signed generator relation')
        group.update(extra)
    words=sorted((key,value) for key,value in group.items() if key)
    rows=_rows(words)
    resolution=min((mask.bit_count() for mask,_ in words),default=0)
    return (rows,resolution) if return_resolution else rows


def sudg(n,p,*,return_resolution=False):
    """Original tabulated useful generators, including saturated designs."""
    if int(n)!=n or int(p)!=p or n<1 or not 0<=p<n:
        raise ValueError('Need integer n>=1 and 0<=p<n')
    n,p=int(n),int(p);q=n-p
    if p==0:
        I=np.empty((0,0),dtype=int);resolution=n+1
    elif p==1:
        I=np.arange(1,n+1)[None,:];resolution=n
    else:
        if n+1==2**q:
            base=[[i+1 for i in range(q) if mask>>i&1] for mask in range(1,2**q) if mask.bit_count()>1]
        elif q==3 and p<=4:
            base=[[1,2],[1,3],[2,3],[1,2,3]]
        elif q==4 and p<=11:
            base=[[1,2,3],[2,3,4],[1,3,4],[1,2,4],[1,2,3,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
        elif q==5 and p<=6:
            base=([[1,2,3],[2,3,4],[3,4,5],[1,3,4],[1,4,5],[2,4,5]] if p==6 else
                  [[1,2,3],[1,2,4],[2,3,4,5]] if p==3 else
                  [[1,2,3,5],[1,2,4,5],[1,3,4,5],[2,3,4,5],[1,2,3,4]])
        elif q==6 and p<=5:
            base=([[1,2,3,4],[3,4,5],[2,4,5,6],[1,4,5,6],[1,2,6]] if p==5 else
                  [[2,3,4,6],[1,3,4,6],[1,2,4,5],[1,2,3,5]] if p==4 else
                  [[1,2,3,4],[1,2,5,6],[2,4,5,6]])
        elif q==7 and p<=4:
            base=([[1,3,4,6,7],[2,3,5,6,7]] if p==2 else
                  [[1,2,3,7],[2,3,4,5],[1,3,4,6],[1,2,3,4,5,6,7]])
        else:
            raise ValueError('Requested design is unavailable in the original generator table')
        words=[(sum(1<<(v-1) for v in row+[q+j+1]),1) for j,row in enumerate(base[:p])]
        I=_rows(words)
        _,resolution=cdr(I,return_resolution=True)
    return (I,resolution) if return_resolution else I


def ffd(n,I=None,*,return_generators=False):
    """Two-level factorial design in standard order, with levels 0 and 1."""
    if int(n)!=n or not 1<=n<=20:
        raise ValueError('n must be an integer in [1,20]')
    n=int(n)
    if I is None or np.size(I)==0:
        I=np.empty((0,0),int)
    elif np.isscalar(I) and not isinstance(I,str):
        I=sudg(n,int(I))
    words=_words(I) if np.size(I) else []
    q=n-len(words)
    if q<1:
        raise ValueError('Need at least one independent factor')
    D=((np.arange(2**q)[:,None]>>np.arange(q))&1).astype(int)
    D=np.column_stack((D,np.zeros((len(D),len(words)),int)))
    for j,(mask,sign) in enumerate(words):
        derived=[i for i in range(q,n) if mask>>i&1]
        if len(derived)!=1:
            raise ValueError('Each generator must include exactly one distinct derived factor')
        base=[i for i in range(q) if mask>>i&1]
        if not base:
            raise ValueError('Each generator needs an independent factor')
        D[:,derived[0]]=(sign*np.prod(2*D[:,base]-1,axis=1)>0)
    return (D,np.asarray(I)) if return_generators else D


def yates(y,*,return_labels=False):
    """Mean, main and interaction effects from responses in standard order."""
    a=np.asarray(y,dtype=float)
    n=len(a)
    if n<1 or n&(n-1):
        raise ValueError('Response count must be a power of two')
    ef=a.copy()
    for _ in range(n.bit_length()-1):
        ef=np.concatenate((ef[1::2]+ef[::2],ef[1::2]-ef[::2]),axis=0)
    ef*=2/n;ef[0]/=2
    labels=cnr2cl(_rows([(mask,1) for mask in range(1,n)]))
    return (ef,labels) if return_labels else ef


def ryates(ef):
    """Inverse Yates transform in standard order."""
    a=np.asarray(ef,dtype=float)
    n=len(a)
    if n<1 or n&(n-1):
        raise ValueError('Effect count must be a power of two')
    k=n.bit_length()-1
    D=2*ffd(k)-1 if k else np.empty((1,0))
    H=np.column_stack((np.ones(n),*[np.prod(D[:,[i for i in range(k) if mask>>i&1]],axis=1) for mask in range(1,n)]))
    coefficients=a.copy();coefficients[1:]/=2
    return H@coefficients


def alias(I0,order=np.inf):
    """Alias classes as lists of signed labels, dropping terms above order."""
    relation=_words(cdr(I0))
    k=max(mask.bit_length() for mask,_ in relation)
    if k>20:
        raise ValueError('Alias enumeration is limited to 20 factors')
    group=[(0,1),*relation]
    unseen=set(range(2**k));classes=[]
    while unseen:
        representative=min(unseen,key=lambda v:(v.bit_count(),v))
        words=[(representative^mask,sign) for mask,sign in group]
        unseen.difference_update(mask for mask,_ in words)
        words=sorted((mask,sign) for mask,sign in words if mask.bit_count()<=order)
        if words:
            classes.append([('I' if mask==0 else cnr2cl(_rows([(mask,sign)]))[0]) for mask,sign in words])
    return classes
