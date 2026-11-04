function ind = findcross(x,v)
%FINDCROSS Pure MATLAB replacement for source/findcross.c.
% One-based index preceding a crossing. Preserves the C plateau convention.
if nargin<2, v=0; end
x=x(:); ind=zeros(0,1);
if any(~isfinite(x)) || ~isscalar(v) || ~isfinite(v)
    error('WAFO:input','Input must be finite and level scalar');
end
s=sign(x-v); first=find(s~=0,1);
if isempty(first), return; end
if first>1, ind(end+1,1)=first-1; end
previous=s(first);
for j=first+1:length(x)
    if s(j)~=0 && s(j)~=previous
        ind(end+1,1)=j-1; previous=s(j);
    end
end
end
