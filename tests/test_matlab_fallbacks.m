function test_matlab_fallbacks
% Run in MATLAB/Octave after adding ../matlab_fallbacks to the path.
% This file has not been executed in the Python-only build environment.
assert(isequal(findcross([0 0 1 0 0 -1 0 1]),[2;5;7]));
assert(isequal(findrfc([0 2 1 3 0],.5),[1;2;3;4]));
assert(isequal(findrfc([0 2 1 3 0],2),[1;4]));
[b,i]=csort([3 -1 3 2]);
assert(isequal(b,[-1 2 3 3])); assert(isequal(i,[2 4 1 3]));
A=complex(zeros(1,16)); A(2)=1+2i; A(3)=3-1i;
w=0:4; k=w.^2/9.81;
[s,d]=disufq(A,w,k,inf,9.81,2,3);
assert(abs(s(3)-2*A(2)^2*k(2))<1e-12);
assert(abs(s(4)-2*A(2)*A(3)*(k(2)+k(3)))<1e-12);
assert(abs(d(2)+2*A(2)*conj(A(3))*(k(3)-k(2)))<1e-12);
assert(norm(disufq(A,w,k,inf,9.81,2,3)-s-d)<1e-12);
L=simduff(200,.001,.1,0,1,2,123456789);
assert(isequal(size(L),[2 3])); assert(all(isfinite(L(:))));
disp('MATLAB fallback smoke tests passed');
end
