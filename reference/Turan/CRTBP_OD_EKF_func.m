%--------------------------------------------------------------------------
%
%        CRTBP Autonomous Orbit Determination application 
%               
%        EKF estimation function
%
%        Use together with CRTBP_OD_EKF_main.m
%
%        By Erdem Turan (Last modified 23/Feb/2023)
%
%--------------------------------------------------------------------------
% This function provides the sequential filter(extended Kalman filter, EKF)
% solution for satellite-to-satellite tracking based autonomous orbit 
% determination for the Circular Restricted Three-body problem (cislunar).
% Estimated state vector consists of position and velocity states of both
% spacecraft (6 position and 6 velocity). 
% 3 Different navigation data types are considered: Satellite-to-satellite 
% tracking (SST) based range, range-rate and angle. If more than one 
% measurement type is requested in the analysis then the system processes 
% all the navigation data at the same time. In this version of this file 
% to keep it simple, any visibility problem is not considered, since 
% based-on the experience, only 1-2% of the measurements are blocked by the 
% Moon if orbits are around different Lagrangian points. Initial S/C states 
% are provided by the database IC_CRTBP.m file. Change Y0_true variable if 
% user defined IC is requested. This file uses CRTBP.m for propagation and 
% STM_CRTBP.m, G_CRTBP.m for state transition matrix calculations.
% Adaptive estimation of process noise covariance matrix is also considered
% in this file. In this case (Adaptive EKF), the forgetting parameter
% (alpha) is used in which 0.4 is found optimal for this problem. User may
% change this value, between 0-1. If alpha = 1, then AEKF turns into EKF.
% User defined process noise matrix can also be used in this script. In
% this case, change the predefined sigm = 1e-12. Measurement noise
% covariance matrix is defined by the known measurement errors, so that
% the navigation system knows exactly the measurement noise. User may
% add errors into the measurement noise covariance matrix (find the line: 
% sigma_range_R). ODE settings can be changed via the options line. In 
% addition, ode113 is known for orbital applications. This file provides
% output of estimation errors (position, velocity), uncertainty (position,
% velocity) and their RSS values. 
%
%
% Inputs:
% P0            Initial State Covariance matrix m and m/s diag(12by12)
% Yerr          Initial State Error m and m/s (1by12)
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
% bias_range        SST range measurement bias in meter 
% bias_rangerate    SST range-rate measurement bias in m/s  
% bias_rel_angle    SST angle measurement bias in arcsec
% 
% Output:
% rsspos        RSS position estimation state vector
% rssvel        RSS velocity estimation state vector
% rssposunc     RSS position uncertainty vector
% rssvelunc     RSS velocity uncertainty vector
% ErrYfull      Full Estimation error vector
% SigUncfull    Full Uncertainty vector
% All mean values are displayed in the end including plots
%
% Note: This script is used to run CRTBP_OD_EKF_main.m
%       

function [rsspos, rssvel, rssposunc, rssvelunc, ErrYfull, SigUncfull] = ...
    CRTBP_OD_EKF_func(P0, Yerr, SC1, SC2, rm, rrm, ram, simdur, ...
    simstep, sigma_range, sigma_range_rate, sigma_rel_angle, ...
    bias_range, bias_rangerate, bias_rel_angle)

global AuxParam

AuxParam.rangemeas      = rm;
AuxParam.rangeratemeas  = rrm;
AuxParam.relanglemeas   = ram;
AuxParam.AdapFilter     = 1;
AuxParam.infowindow     = 0;

LU=384747963;   %m, lenght unit
SU=1.025351467559547e+03; %m/s, velocity convertion

sigma_range = (1/LU)*sigma_range;  %range measurement in LU
sigma_range_rate = (1/SU)*sigma_range_rate; % range-rate measurement in SU
sigma_rel_angle = sigma_rel_angle;

% Measurement noise covariance matrix sigma values (no error is assumed)
sigma_range_R = sigma_range;
sigma_range_rate_R = sigma_range_rate;
sigma_rel_angle_R = sigma_rel_angle;

%step size in TU
step=2.6667e-06*simstep;

%load initial states
IC_CRTBP
 
%simulation duration in days
daysnum=simdur;
n_obs = (daysnum/4.343)/step;

%9.752753388846403e-07 = 1mm/s 
%2.599104078947392e-06 = 1km
%384747.963km = 1LU
%1/4.343day   = 1TU
%1.025351467559547e+03 conversion parameter (LU/TU to m/s)

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
Yerr = Yerr.*[ones(1,3)*(1/LU) ones(1,3)*(1/SU) ones(1,3)*(1/LU) ...
        ones(1,3)*(1/SU)];
Y = Y0_true+Yerr;

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

% Initial Measurement Covariance
sigm = 1e-12;
delt = step;
Qdt1=[((delt^4)*sigm)/3 0 0 ((delt^3)*sigm)/2 0 0;
        0 ((delt^4)*sigm)/3 0 0 ((delt^3)*sigm)/2 0
        0 0 ((delt^4)*sigm)/3 0 0 ((delt^3)*sigm)/2
        ((delt^3)*sigm)/2 0 0 ((delt^2)*sigm) 0 0
        0 ((delt^3)*sigm)/2 0 0 ((delt^2)*sigm) 0
        0 0 ((delt^3)*sigm)/2 0 0 ((delt^2)*sigm)];
    
Q=[Qdt1 zeros(6,6); zeros(6,6) Qdt1];

%Adaptive parameter (forgetting parameter)
if AuxParam.AdapFilter == 1
    alpha=0.4;
else
    alpha=1;
end

% Measurement biases
bias_range=bias_range;
bias_rangerate=bias_rangerate;
bias_rel_angle=bias_rel_angle;

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

    %Estimated trajectory
    % 1st S/C
    [~,xx]=ode113('CRTBP',[t_old t], Y(1:6)',options,[],1,mu);
    Y1=xx(end,:); 
    % 2nd S/C
    [~,xxx]=ode113('CRTBP',[t_old t], Y(7:12)',options,[],1,mu);
    Y2=xxx(end,:);
    % Combined State
    Y=[Y1 Y2]';

    %True trajectory
    % 1st S/C
    [~,xx]=ode113('CRTBP',[t_old t], Y0_true(1:6)',options,[],1,mu);
    Y1=xx(end,:); 
    % 2nd S/C
    [~,xxx]=ode113('CRTBP',[t_old t], Y0_true(7:12)',options,[],1,mu);
    Y2=xxx(end,:);
    % Combined State
    Y0_true=[Y1 Y2]';
    
    % State-Transition Matrix
    % 1st S/C
    STM=STM_CRTBP(t_old,t,Y_old(1:6),mu);
    % 2nd S/C
    STM1=STM_CRTBP(t_old,t,Y_old(7:12),mu);
    % Combined STM
    Phi=[STM, zeros(6,6); zeros(6,6), STM1];

    %Time Update: Covariance
    P = Phi*P*Phi'+Q;

    % Measurements from models and their partials
    dr        = Y(1:3)-Y(7:9); 
    [udr rho] = unit(dr);
    %Range
    if AuxParam.rangemeas == 1
        %measurement
        obs_range = norm(Y0_true(1:3)-Y0_true(7:9))+...
            normrnd(bias_range,sigma_range);
        %range from model
        rangeHat = norm(dr);
        Hrr = shiftdim(udr,-1);
        Hrv = zeros(size(Hrr));
        %Range measurement partials
        Hr   = [Hrr Hrv]; 
    end

    %Range-rate 
    if AuxParam.rangeratemeas == 1
        %measurement
        rau_true=Y0_true(1:3)-Y0_true(7:9);
        Dist_true=norm(rau_true);
        raudot_true=Y0_true(4:6)-Y0_true(10:12);
        obs_rrate=(dot(raudot_true,rau_true)/Dist_true)+...
            normrnd(bias_rangerate,sigma_range_rate);
        %range-rate from model
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
        %measurements
        rau_true=Y0_true(1:3)-Y0_true(7:9);
        Dist_true=norm(rau_true);
        obs_az = atan2(Y0_true(2)-Y0_true(8),Y0_true(1)-Y0_true(7))+...
            normrnd(bias_rel_angle,sigma_rel_angle);
        obs_el = atan2(Y0_true(3)-Y0_true(9),Dist_true)+...
            normrnd(bias_rel_angle,sigma_rel_angle);

        %angle from model
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
        d=(obs_range - rangeHat);
        H=Hr;
        H=[H -H];
        R=[sigma_range_R.^2];
    end
    if AuxParam.rangeratemeas  == 1
        d=[d;(obs_rrate - rrate)];
        H=[H;Hv -Hv];
        R=[R sigma_range_rate_R.^2];
    end
    if AuxParam.relanglemeas == 1
        d=[d;az-obs_az; el-obs_el];
        H=[H; Hang3];
        R=[R sigma_rel_angle_R^2 sigma_rel_angle_R^2];
    end
    R = diag(R);

    % Measurement update
    K = P*H'*inv(R+H*P*H');

    % Correction
    Y = Y + K*d;

    % Covariance update
    P = (eye(12)-K*H)*P;

    % Update Q
    Q = alpha*Q + (1-alpha)*(K*d*d'*K');

    % Estimation Errors
    ErrY = Y-Y0_true;
    Sigma = zeros(12,1);
    
    for ii=1:12
        Sigma(ii) = sqrt(P(ii,ii));
    end

    %RSS Pos and Vel errors
    rsspos(k,:)=rssq([ErrY(1:3); ErrY(7:9)]);
    rssvel(k,:)=rssq([ErrY(4:6); ErrY(10:12)]);
    rssposunc(k,:)=rssq([Sigma(1:3); Sigma(7:9)]);
    rssvelunc(k,:)=rssq([Sigma(4:6); Sigma(10:12)]);

    % Full history
    ErrYfull(k,:) = ErrY;
    SigUncfull(k,:) = Sigma;

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


