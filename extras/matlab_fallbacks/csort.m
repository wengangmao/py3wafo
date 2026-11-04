function [b,ind] = csort(a,amin,amax,varargin)
%CSORT Portable stable replacement for the C counting sort MEX.
% Uses MATLAB SORT (O(n log n)) and preserves equal-key order.
if ~isvector(a) || isempty(a) || any(~isfinite(a(:))) || any(a(:)~=fix(a(:)))
    error('WAFO:input','Expected a nonempty vector of finite integers');
end
if nargin>=2 && ~isempty(amin) && any(a(:)<amin)
    error('WAFO:input','Value below amin');
end
if nargin>=3 && ~isempty(amax) && any(a(:)>amax)
    error('WAFO:input','Value above amax');
end
[~,p]=sortrows([a(:),(1:numel(a))'],[1 2]);
b=reshape(a(p),size(a)); ind=reshape(p,size(a));
end
