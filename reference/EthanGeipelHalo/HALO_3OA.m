function InitialConditions = HALO_3OA(mu, Ax, Az, ne) % Ax and Az is in km
    %% Get L points
    [L1, L2, ~] = getLpoints(mu);

    %% Analytic variables For Richardson Analytic Construction
    syms lam
    au = 3.84747963e5; % km
    re = au-L1*au;
    
    gam(1) = ((1-mu)-L1) / (1-mu-mu);
    gam(2) = -((1-mu)-L2) / (1-mu-mu);
    gam(3) = ((1-mu)-0) / (1-mu-mu);

    c = zeros(4,3);
    for n = 2:1:4
        c(n,1) = (1/(gam(1)^3))*( (1)^n*mu +  (-1)^n * (((1-mu)*gam(1)^(n+1))/((1-gam(1))^(n+1))) );
        c(n,2) = (1/(gam(2)^3))*( (-1)^n*mu + (-1)^n * (((1-mu)*gam(2)^(n+1))/((1+gam(2))^(n+1))) );
        c(n,3) = (1/(gam(3)^3))*( 1 - mu + ((mu*gam(3)^(n+1)))/((1+gam(3))^(n+1)) );
    end
    eqn = lam^4 + (c(2,1)-2)*lam^2-(c(2,1)-1)*(1+2*c(2,1)) == 0;
    lambda(:,1) = vpasolve(eqn, lam);
    eqn = lam^4 + (c(2,2)-2)*lam^2-(c(2,2)-1)*(1+2*c(2,2)) == 0;
    lambda(:,2) = vpasolve(eqn, lam);
    eqn = lam^4 + (c(2,3)-2)*lam^2-(c(2,3)-1)*(1+2*c(2,3)) == 0;
    lambda(:,3) = vpasolve(eqn, lam);
    lambda = lambda(2,:);

    del = lambda(1,:).^2 - c(2,:);
    del(3) = 2.66029*10^-6;

    k = 2.*lambda(1,:)./((lambda(1,:).^2+1-c(2,:)));

    d1 = 3*lambda(1,:).^2./k(1,:).*(k(1,:).*(6.*lambda(1,:).^2-1)-2.*lambda(1,:));
    d2 = 8*lambda(1,:).^2./k(1,:).*(k(1,:).*(11.*lambda(1,:).^2-1)-2.*lambda(1,:));

    a21 = 3.*c(3,:).*(k.^2-2)./(4.*(1+2.*c(2,:)));
    a22 = 3.*c(3,:)./(4*(1+2*c(2,:)));
    a23 = -(3*c(3,:).*lambda(1,:)./(4.*k(1,:).*d1(1,:))).*(3*k(1,:).^3.*lambda(1,:)-6.*k(1,:).*(k(1,:)-lambda(1,:))+4);
    a24 = -3.*c(3,:).*lambda(1,:)./(4.*k(1,:).*d1(1,:)).*(2+3.*k(1,:).*lambda(1,:));

    b21 = -3.*c(3,:).*lambda(1,:)./2./d1(1,:).*(3.*k(1,:).*lambda(1,:)-4);
    b22 = 3.*c(3,:).*lambda(1,:)./d1(1,:);

    d21 = -c(3,:)./2./(lambda(1,:).^2);
    c31 = 3./64./(lambda(1,:).^2).*(4.*c(3,:).*a24(1,:) + c(4,:));

    a31 = -9.*lambda(1,:)./(4.*d2(1,:)).*( 4.*c(3,:).*(k(1,:).*a23(1,:)-b21(1,:))+ k(1,:).*c(4,:).*(4+k(1,:).^2) ) ...
        +((9.*lambda(1,:).^2 + 1 - c(2,:))./2./d2(1,:)).*( 3.*c(3,:).*(2*a23(1,:)-k(1,:).*b21(1,:)) +c(4,:).*(2+3.*k(1,:).^2) );
    a32 = -(1./d2(1,:)).*( (9.*lambda(1,:)./4).*( 4.*c(3,:).*(k(1,:).*a24(1,:) - b22(1,:)) + k(1,:).*c(4,:) ) ...
        +(3/2).*(9.*lambda(1,:).^2 + 1 - c(2,:)).*( c(3,:).*(k(1,:).*b22(1,:) + d21(1,:) - 2.*a24(1,:)) - c(4,:) ));

    b31 = 3./8./d2(1,:).*( 8.*lambda(1,:).*( 3.*c(3,:).*(k(1,:).*b21(1,:)-2.*a23(1,:)) - c(4,:).*(2+3.*k(1,:).^2) ) ...
        +(9.*lambda(1,:).^2 + 1 + 2.*c(2,:)).*( 4.*c(3,:).*(k(1,:).*a23(1,:) - b21(1,:)) + k(1,:).*c(4,:).*(4+k(1,:).^2) )  );
    b32 = (1./d2(1,:)).*( 9.*lambda(1,:).*( c(3,:).*(k(1,:).*b22(1,:) + d21(1,:) - 2.*a24(1,:)) - c(4,:) )...
        +(3/8).*(9.*lambda(1,:).^2 + 1 + 2.*c(2,:)).*(4.*c(3,:).*(k(1,:).*a24(1,:) - b22(1,:)) + k(1,:).*c(4,:) )   );

    a1 = -(3/2).*c(3,:).*(2.*a21(1,:) + a23(1,:) + 5.*d21(1,:)) - (3/8).*c(4,:).*(12-k(1,:).^2);
    a2 = (3/2).*c(3,:).*(a24 - 2.*a22(1,:)) + (9/8).*c(4,:);
    a1(3) = -1.25889*10^-5;
    a2(3) = 1.43702*10^-6;

    d31 = 3./64./(lambda(1,:).^2).*(4.*c(3,:).*a24(1,:) + c(4,:));
    d32 = 3./64./(lambda(1,:).^2).*( 4.*c(3,:).*(a23(1,:) - d21(1,:)) + c(4,:).*(4+k(1,:).^2) );

    s1 = 1./(2.*lambda(1,:).*(lambda(1,:).*(1+k(1,:).^2) - 2.*k(1,:))).*( ...
        (3/2).*c(3,:).*( 2.*a21(1,:).*(k(1,:).^2-2) - (a23(1,:).*(k(1,:).^2 + 2)) - 2.*k(1,:).*b21(1,:) )  ...
        -(3/8).*c(4,:).*(3.*k(1,:).^4-8.*k(1,:).^2 + 8));
    s2 = 1./(2.*lambda(1,:).*(lambda(1,:).*(1+k(1,:).^2) - 2.*k(1,:))).*( ...
        (3/2).*c(3,:).*( 2.*a22(1,:).*(k(1,:).^2-2) + (a24(1,:).*(k(1,:).^2 + 2)) + 2.*k(1,:).*b22(1,:) + 5.*d21(1,:))  ...
        +(3/8).*c(4,:).*(12- k(1,:).^2) );
    s1(3) = -1.59141*10^-6;
    s2(3) = 6.29433*10^-6;

    l1 = a1(1,:) + 2.*lambda(1,:).^2.*s1(1,:);
    l2 = a2(1,:) + 2.*lambda(1,:).^2.*s2(1,:);

    Ax_min = sqrt(abs(del(1,:)./l1(1,:)));
    Azt = sqrt((-del(1,:) - l1(1,:).*(Ax_min*re).^2)./l2(1,:));

    %% Construct the Path
    Lpoint = 1;
    Az = Az/re;
    Ax = -Ax/re;
    if (Lpoint == 1) 
        Ay = k(1)*Ax; 
    end
    if Lpoint == 2
        Ax = Ax_min(2); Ay = 0;
    end

    PHI = 0; PSI = [PHI + pi/2; PHI + 3*pi/2];
    w = 1 + s1(1,:).*(Ax^2) + s2(1,:).*(Az^2);

    %% Iterate Orbit Trace
    x = zeros(1,3); y = zeros(1,3); z = zeros(1,3);
    dx = zeros(1,3); dy = zeros(1,3); dz = zeros(1,3); 
    i = 0;
    for t = 0:5*24*60*60/.2:180*24*60*60
        i = i + 1;
        % Time non-dimensionalizing
        s = ne*t; 
        tau = w*s;
        tau1 = lambda(1,:).*tau + PHI;
    
        x(i,:) = a21(1,:).*Ax.^2 + a22(1,:).*Az.^2 - Ax.*cos(tau1) + (a23(1,:).*Ax.^2 - a24(1,:).*Az.^2).*cos(2.*tau1) ...
            + (a31(1,:).*Ax.^3 - a32(1,:).*Ax.*Az.^2).*cos(3.*tau1);

        y(i,:) = k(1,:).*Ax.*sin(tau1) + (b21(1,:).*Ax^2 - b22(1,:).*Az^2).*sin(2.*tau1) ...
            + (b31(1,:).*Ax^3 - b32(1,:).*Ax.*Az^2).*sin(3.*tau1);

        deln = 2 - 3;
        z(i,:) = deln.*Az.*cos(tau1) + deln.*d21(1,:).*Ax.*Az.*(cos(2.*tau1) - 3) ...
            + deln.*(d32.*Az.*Ax^2 - d31(1,:).*Az^3).*cos(3.*tau1);

        dx(i,:) =  -Ax.*sin(tau1) + -2*(a23(1,:).*Ax.^2 - a24(1,:).*Az.^2).*sin(2.*tau1) ...
            + -3*(a31(1,:).*Ax.^3 - a32(1,:).*Ax.*Az.^2).*sin(3.*tau1);

        dy(i,:) = k(1,:).*Ax.*lambda(1,:).*cos(tau1) + 2*(b21(1,:).*Ax^2 - b22(1,:).*Az^2).*lambda(1,:).*cos(2.*tau1) ...
            + 3*(b31(1,:).*Ax^3 - b32(1,:).*Ax.*Az^2).*lambda(1,:).*cos(3.*tau1);

        dz(i,:) = -deln.*Az.*sin(tau1) + -2*deln.*d21(1,:).*Ax.*Az.*(sin(2.*tau1)) ...
            + -3*deln.*(d32.*Az.*Ax^2 - d31(1,:).*Az^3).*sin(3.*tau1);

    end

    %% Plotting
    Plotter = 0;
    if Plotter == 1
        figure(8)
        plot(dx(:,1)/max(dx(:,1))); hold on
        plot(dy(:,1)/max(dy(:,1))); plot(dz(:,1)/max(dz(:,1)))
        legend(['dx'; 'dy'; 'dz'])

        figure(4)
        plot3(au*L1+re*x(:,Lpoint),re*y(:,Lpoint),re*z(:,Lpoint), 'k' )
        hold on; axis square; grid on
        plot(au*(1-mu),(0),'b*','LineWidth',4);
        plot(au*L1,0,'kx')

        figure(2)
        plot(re*x(:,Lpoint),re*z(:,Lpoint), 'k')
        hold on; axis square; grid on
        plot(re*(1-mu),(0),'b*','LineWidth',4);

        figure(3)
        plot(re*x(:,Lpoint),re*y(:,Lpoint), 'k')
        hold on; axis square; grid on
        plot((1-mu),(0),'b*','LineWidth',4); plot((-mu),(0),'b*','LineWidth',4)

        figure(1) % y-z
        plot(y(:,Lpoint),z(:,Lpoint),'k')
        hold on; axis square; grid on
        plot((-mu),(0),'b*','LineWidth',4)

        % General Plot Lims (If wanted)
        % axisdiv = 5*10^5;
        % xlim([-2*axisdiv 2*axisdiv]); xticks([-2*axisdiv -axisdiv 0 axisdiv 2*axisdiv])
        % ylim([-2*axisdiv 2*axisdiv]); yticks([-2*axisdiv -axisdiv 0 axisdiv 2*axisdiv])
    end
    
    %% Return Prints
    fprintf('x0 = %3.3f\n', x(1)*re)
    fprintf('z0 = %3.3f\n', z(1)*re)
    fprintf('dy0= %3.3f\n', dy(1)*re*ne)
    fprintf('Finished 3rd Order Approximation!\n')

    InitialConditions = [x(1)*re; 0; z(1)*re; 0; dy(1)*re*ne; 0];
    
end