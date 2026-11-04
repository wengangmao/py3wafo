function ind = findrfc(tp,h)
%FINDRFC Pure MATLAB port of source/findrfc.c, including terminal zero padding.
if ~isscalar(h) || ~isfinite(h) || h<=0 || any(~isfinite(tp(:)))
    error('WAFO:input','Require finite turning points and positive h');
end
a=tp(:); ind=zeros(0,1); n=length(a);
if n<2, return; end
offset=double(a(1)>a(2)); y=[a(1+offset:end);0];
nc=floor((n-offset)/2);
if length(y)>=3 && ((y(1)>y(2) && y(2)>y(3)) || (y(1)<y(2) && y(2)<y(3)))
    error('WAFO:input','Input is not a turning point sequence');
end
for i=0:nc-1
    mi=2*i+1; pl=2*i+3; xm=y(mi); xp=y(pl); peak=y(2*i+2);
    j=i-1;
    while j>=0 && y(2*j+2)<=peak
        if y(2*j+1)<xm, xm=y(2*j+1); mi=2*j+1; end
        j=j-1;
    end
    if xm<xp
        j=i+1;
        while j<nc && y(2*j+2)<peak
            if y(2*j+3)<=xp, xp=y(2*j+3); pl=2*j+3; end
            j=j+1;
        end
        if j<nc && xp>xm
            if peak-xp>=h, ind=[ind;offset+2*i+2;offset+pl]; end
            continue;
        end
    end
    if peak-xm>=h, ind=[ind;offset+mi;offset+2*i+2]; end
end
ind=sort(ind);
end
