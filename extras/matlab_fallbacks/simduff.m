function L = simduff(number_of_points,dt,z,a,b,alf,seed)
%SIMDUFF Script-function replacement for source/simduff.f.
% Input matches the values formerly written to simduff.in. Returns rows
% [i*dt,x,xdot] at every 100th internal step; no executable or file IPC.
% Park-Miller RNG and Box-Muller/stable formulas follow the original source.
% Double precision replaces the legacy single-precision Fortran arithmetic.
if nargin<7, seed=123456789; end
if number_of_points<2 || dt<=0 || z<=0 || z>=1 || alf<=0 || alf>2
    error('WAFO:input','Invalid oscillator parameters');
end
if seed<=0 || seed>=2147483647 || seed~=floor(seed)
    error('WAFO:input','Seed must be an integer in (0,2147483647)');
end
L=zeros(floor(number_of_points/100),3);
[x,seed]=draw(seed,alf); [vel,seed]=draw(seed,alf);
if a>.01, x=x/sqrt(a); end
s=sqrt(4*z)*dt^(1/alf); k=0;
for i=2:number_of_points
    [noise,seed]=draw(seed,alf);
    xnew=x+vel*dt;
    vel=vel+(-b*x-a*x^3-2*z*vel)*dt+s*noise;
    x=xnew;
    if ~isfinite(x) || ~isfinite(vel), error('WAFO:diverged','Reduce time step'); end
    if mod(i,100)==0
        k=k+1; L(k,:)=[i*dt,x,vel];
    end
end
L=L(1:k,:);
end

function [value,seed]=draw(seed,alf)
seed=mod(16807*seed,2147483647); u1=seed/2147483647;
seed=mod(16807*seed,2147483647); u2=seed/2147483647;
if alf>1.99999
    value=sqrt(-2*log(u1))*cos(2*pi*u2);
else
    v=pi*(u1-.5); xk=alf*v; w=-log(u2)/cos(v-xk);
    value=sin(xk)*exp(-log(w*cos(v))/alf)*w/sqrt(2);
end
end
