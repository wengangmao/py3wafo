function [svec,dvec] = disufq(A,w,kw,h,g,nmin,nmax)
%DISUFQ Pure MATLAB port of source/disufq.c, finite/deep water branches.
% nmin/nmax remain MATLAB one-based inclusive indices. One output gives sum+difference.
w=w(:); kw=kw(:);
if h<=0 || g<=0 || nmin<1 || nmax>min(length(w),length(kw)) || nmin>nmax
    error('WAFO:input','Invalid parameters');
end
if 2*(nmax-1)+1>size(A,2)
    error('WAFO:input','FFT buffer too small for sum frequencies');
end
if any(w(nmin:nmax)<=0) || any(kw(nmin:nmax)<=0)
    error('WAFO:input','Active frequencies must be positive');
end
svec=complex(zeros(size(A))); dvec=svec;
for i=nmin:nmax
    k1=kw(i); w1=w(i);
    if h>10000
        ep=k1; ed=0;
    else
        th=tanh(k1*h); cg=.5*g*(th+k1*h*(1-th*th))/w1;
        t1=.5*g*(k1/w1)^2; t2=.5*w1*w1/g; t3=g*k1/(w1*cg);
        if k1*h<300, t4=k1/sinh(2*k1*h); else, t4=0; end
        ed=(t1-t2+t3)/(1-g*h/cg^2)-t4;
        ep=3*(t1-t2)/(1-t1/k1*tanh(2*k1*h))+3*t2-t1;
    end
    svec(:,2*i-1)=svec(:,2*i-1)+2*A(:,i).^2*ep;
    dvec(:,1)=dvec(:,1)+2*abs(A(:,i)).^2*ed;
    for j=i+1:nmax
        k2=kw(j); w2=w(j);
        if h>10000
            ep=.5*(k1+k2); ed=-.5*(k2-k1);
        else
            if w1==w2, error('WAFO:input','Frequencies must be distinct'); end
            t1=g*(k1/w1)*(k2/w2);
            t2=.5/g*(w1*w1+w2*w2+w1*w2);
            t3=.5*g*(w1*k2*k2+w2*k1*k1)/(w1*w2*(w1+w2));
            t4=1-g*(k1+k2)/(w1+w2)^2*tanh((k1+k2)*h);
            ep=(t1-t2+t3)/t4+t2-.5*t1;
            t2=.5/g*(w1*w1+w2*w2-w1*w2);
            t3=-.5*g*(w1*k2*k2-w2*k1*k1)/(w1*w2*(w1-w2));
            t4=1-g*(k1-k2)/(w1-w2)^2*tanh((k1-k2)*h);
            ed=(t1-t2+t3)/t4+t2-.5*t1;
        end
        svec(:,i+j-1)=svec(:,i+j-1)+4*A(:,i).*A(:,j)*ep;
        dvec(:,j-i+1)=dvec(:,j-i+1)+4*A(:,i).*conj(A(:,j))*ed;
    end
end
if nargout<2, svec=svec+dvec; end
end
