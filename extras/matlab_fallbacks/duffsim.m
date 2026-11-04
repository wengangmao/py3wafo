function L = duffsim(T,dt,z,a,b,alf)
%DUFFSIM Pure MATLAB replacement for the executable-launching WAFO wrapper.
if nargin<5, b=-1; end
if nargin<6, alf=2; end
if nargin<4, a=0; b=1; end
if T<0 || dt<=0, error('WAFO:input','Require T>=0 and dt>0'); end
N=(floor(T/dt)+1)*100;
if N>5000000, error('WAFO:input','Too many internal steps'); end
seed=floor(1e8+rand*899999999);
L=simduff(N,dt*.01,z,a,b,alf,seed);
end
