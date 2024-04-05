function [u,l] = unit(v)
[m,n] = size(v);
if m==1 && n==1
    l = v;
    u = 1;
elseif xor(m==1,n==1) 
    l = norm(v);
    u = v/l;
else 
    l = sqrt(sum(v.^2));
    u = bsxfun(@times,v,1./l); 
end
