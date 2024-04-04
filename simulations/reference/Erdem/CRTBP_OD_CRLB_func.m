%--------------------------------------------------------------------------
%
%               CRTBP Autonomous Orbit Determination application 
%               
%               CRLB estimation function
%
%               Use together with CRTBP_OD_CRLB_main.m 
%
%               By Erdem Turan (Last modified 22/Feb/2023)
%
%--------------------------------------------------------------------------
% This function provides the Cramer-Rao Lower Bound (CRLB) analysis for
% sequential filters. CRLB has been calculated based on inverse of the
% Fisher Information Matrix (FIM). 3 Different navigation data types are
% considered: Satellite-to-satellite tracking (SST) based range, range-rate
% and angle. If more than one measurement type is requested in the analysis 
% then the system processes all the measurements types at the same time.
% In this version of this file to keep it simple, any visibility 
% problem is not considered, since based-on the simulations, only 1-2% of 
% the measurements are blocked by the Moon if orbits are around different 
% Lagrangian points. Initial S/C states are provided by the database 
% IC_CRTBP.m file. Change Y0_true variable if user defined IC is requested.
% This file uses CRTBP.m for propagation and STM_CRTBP.m, G_CRTBP.m for
% state transition matrix calculations. 
% 
%
% Inputs:
% P0            Initial State Covariance matrix m and m/s diag(12by12)
% SC1           First S/C initial states from the database (1-16)
% SC2           Second S/C initial states from the database (1-16)
% simdur        Simulation duration in days
% simstep       Simluation time step in seconds 
% rm            Satellite-to-satellite range measurement logical 1 or 0 
% rrm           Satellite-to-satellite rangerate measurement logical 1 or 0 
% ram           Satellite-to-satellite angle measurement logical 1 or 0 
% sigma_range   SST range measurement error in meter (1 sigma)
% sigma_range_rate  SST range-rate measurement error in m/s (1 sigma)
% sigma_rel_angle   SST angle measurement error in arcsec (az and el)
% 
% Output:
% rssCRLB       RSS position uncertainty in non-dim unit then given in 
%               meter (1sigma). Values are considered 3 position states of 
%               both S/C in the end RSS values are calculated based on 
%               6 states
%               
% rssCRLBvel    RSS velocity uncertainty in non-dim unit then given in 
%               m/s (1sigma). Values are considered 3 velocity states of 
%               both S/C, in the end RSS values are calculated based on 
%               6 states
% fullCRLB      Full uncertainty history (12by:)

% All mean values are displayed in the end including plots
%
% Note: This script is intended to use together with CRTBP_OD_CRLB_main.m
% if live info is requested about the remaining simulation time then make
% AuxParam.infowindow == 1. 
%       
%% 

function [rssCRLB, rssCRLBvel, fullCRLB] = CRTBP_OD_CRLB_func(P0, SC1, ...
    SC2, rm, rrm, ram, simdur, simstep, sigma_range, sigma_range_rate, ...
    sigma_rel_angle) 

global AuxParam

AuxParam.rangemeas      = rm;
AuxParam.rangeratemeas  = rrm;
AuxParam.relanglemeas   = ram;
AuxParam.CRLB           = 1;
AuxParam.infowindow     = 0;

LU=384747963;   %m, lenght unit
SU=1.025351467559547e+03; %m/s, velocity convertion

% Useful conversion parameters
%9.752753388846403e-07 = 1mm/s 
%2.599104078947392e-06 = 1km
%384747.963km = 1LU
%1/4.343day   = 1TU
%1.025351467559547e+03 conversion parameter (LU/TU to m/s)

sigma_range = (1/LU)*sigma_range;  %range measurement in LU
sigma_range_rate = (1/SU)*sigma_range_rate; % range-rate measurement in SU
sigma_rel_angle = sigma_rel_angle;

%step size in TU
step=2.6667e-06*simstep;

%load initial states
IC_CRTBP
 
%simulation duration in days
daysnum=simdur;
n_obs = (daysnum/4.343)/step;

% 1:EML2south, 2:EML2south,3:EML2north, 4:EML2north 5:EML1south 6:EML1south
% 7:EML1north  8:EML1north 9:Lunar Elliptic 10:Lunar Polar-Circular
% 11:NRHO EML2south, 12:NRHO EML1south 13:NRHO EML2north, 
% 14:NRHO EML1north 15:EML2 Lyapunov 16:EML1 Lyapunov

Sc_1st=SC1;
Sc_2nd=SC2;

% Initial States
Y0_true = [IC(Sc_1st,:) IC(Sc_2nd,:)]';

%Gravational constant
G=1;
mu=0.012155650403207;

% Initialization
t = 0;

%Initial state 
Y = Y0_true;

%Initial covariance matrix
P = zeros(12);
for i= 1:3
    P(i,i) = P0(i)*((1/LU)^2);
    P(i+6,i+6) = P0(i)*((1/LU)^2);
end
for i=4:6
    P(i,i) = P0(i)*((1/SU)^2);
    P(i+6,i+6) = P0(i)*((1/SU)^2);
end

% Initial FIM
J = inv(P);

k=0;

% Information window
if AuxParam.infowindow == 1
    f = waitbar(0,'...','Name','Please wait',...
        'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
    setappdata(f,'canceling',0);
end

% ODE option
options=odeset('RelTol',1e-12,'AbsTol',1e-22);

% Measurement Loop
for i=1:n_obs

    k=k+1;
    % Previous step
    t_old = t;
    Y_old = Y;

    % Propagation to measurement epoch
    t = i*step;     % Time since epoch

    %numerical integration
    % 1st S/C
    [~,xx]=ode113('CRTBP',[t_old t], Y(1:6)',options,[],1,mu);
    Y1=xx(end,:); 
    % 2nd S/C
    [~,xxx]=ode113('CRTBP',[t_old t], Y(7:12)',options,[],1,mu);
    Y2=xxx(end,:);
    % Combined State
    Y=[Y1 Y2]';
    
    % State-Transition Matrix
    % 1st S/C
    STM=STM_CRTBP(t_old,t,Y_old(1:6),mu);
    % 2nd S/C
    STM1=STM_CRTBP(t_old,t,Y_old(7:12),mu);
    % Combined STM
    Phi=[STM, zeros(6,6); zeros(6,6), STM1];

    % Measurements from models and their partials
    dr        = Y(1:3)-Y(7:9); 
    [udr rho] = unit(dr);
    %Range
    if AuxParam.rangemeas == 1
        rangeHat = norm(dr);
        Hrr = shiftdim(udr,-1);
        Hrv = zeros(size(Hrr));
        %Range measurement partials
        Hr   = [Hrr Hrv]; 
    end

    %Range-rate 
    if AuxParam.rangeratemeas == 1
        dr        = Y(1:3)-Y(7:9); 
        rangeHat = norm(dr);
        rau=[Y(1)-Y(7) Y(2)-Y(8) Y(3)-Y(9)];
        raudot=[Y(4)-Y(10) Y(5)-Y(11) Y(6)-Y(12)];
        rrate=dot(raudot,rau)/rangeHat;
        dv        = Y(4:6)-Y(10:12); 
        Hvr = shiftdim(dv.*repmat(1./rho,3,1) - ...
            dr.*repmat(dot(dr,dv)./rho.^3,3,1),-1); 
        Hvv = shiftdim(udr,-1);
        %Range-rate measurement partials
        Hv   = [Hvr Hvv]; 
    end

    if AuxParam.relanglemeas == 1
        %S/C to S/C angle measurements
        dr = Y(1:3) - Y(7:9);
        rho = norm(dr);
        denum = (rho.^3)*sqrt(1-(dr(3).^2 / rho.^2));
        Har(1,1) = dr(2)/(rho.^2);
        Har(1,2) = -dr(1)/(rho.^2);
        Har(2,1) = dr(1)*dr(3)/denum;
        Har(2,2) = dr(2)*dr(3)/denum;
        Har(2,3) = (dr(3).^2-rho.^2)/denum;
        Har(1,3:6) = zeros;
        Har(2,4:6) = zeros;
        Hang3 = [Har -Har];
    end

    % Measurement partials and measurement covariance matrices
    H=[];
    R=[];
    if AuxParam.rangemeas == 1
        H=Hr;
        H=[H -H];
        R=[sigma_range.^2];
    end
    if AuxParam.rangeratemeas  == 1
        H=[H;Hv -Hv];
        R=[R sigma_range_rate.^2];
    end
    if AuxParam.relanglemeas == 1
        H=[H; Hang3];
        R=[R sigma_rel_angle^2 sigma_rel_angle^2];
    end
    R = diag(R);

    % Fisher Information Matrix
    if isempty(R)
        J3 = Phi/J;
        J4 = J3*Phi';
        J = inv(J4);
    else
        J1 = H'/R;
        J2 = J1*H;
        J3 = Phi/J;
        J4 = J3*Phi';
        J = inv(J4) + J2;
    end

    % P >= inv(J)
    P_CRLB = inv(J);

    if AuxParam.CRLB == 1
        CRLB = zeros(12,1);
        for ii=1:12
            CRLB(ii) = sqrt((P_CRLB(ii,ii)));
        end
    end

    %Pos and Vel Uncertainty
    rssCRLB(k,:)=rssq([CRLB(1:3); CRLB(7:9)]);
    rssCRLBvel(k,:)=rssq([CRLB(4:6); CRLB(10:12)]);

    fullCRLB(k,:)=CRLB;

    if any(i == [0:100:n_obs])
        formatSpec = 'Simulation is running: %4.2f ';
        disp([sprintf(formatSpec,(i/n_obs)*100) '%'])

    if AuxParam.infowindow == 1
        if getappdata(f,'canceling')
            break
        end
            
        % Update waitbar and message
        waitbar((i/n_obs),f,sprintf('Simulation is running: %4.2f %%',...
            (i/n_obs)*100))
    end  
    end

end

if AuxParam.infowindow == 1
    delete(f)
end

end


