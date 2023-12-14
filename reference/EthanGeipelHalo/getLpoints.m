function [L1, L2, L3] = getLpoints(mu)

    syms func x y z 
    R1 = sqrt((x+mu)^2 + y^2 + z^2);
    R2 = sqrt((x+mu-1)^2 + y^2 + z^2);
    func = (1/2)*(x^2+y^2) + (1-mu)/R1 + mu/R2;
    dx = diff(func, x);
    yl = 0; zl = 0;
    dx_l = subs(dx,   y, yl);
    dx_l = subs(dx_l, z, zl);

    L1 = double( vpasolve( dx_l == 0,  0) );
    L2 = double( vpasolve( dx_l == 0,  1) );
    L3 = double( vpasolve( dx_l == 0, -1) );

end