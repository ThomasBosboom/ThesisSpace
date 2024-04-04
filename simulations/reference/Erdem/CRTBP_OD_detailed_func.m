%--------------------------------------------------------------------------
%
%        CRTBP Autonomous Orbit Determination application 
%               
%        Detailed filter function (EKF, AEKF, AIEKF, CKF, etc.)
%        Triple S/C formation included (mesh or centralized topology)
%        Clock parameters included (neglected, estimated, considered)
%
%        Use together with CRTBP_OD_detailed_main.m
%
%        By Erdem Turan (Last modified 28/Feb/2023)
%
%--------------------------------------------------------------------------
% This function provides the sequential filter(extended Kalman filter, EKF)
% solution for satellite-to-satellite tracking based autonomous orbit 
% determination for the Circular Restricted Three-body problem (cislunar).
% Estimated state vector consists of position and velocity states of both
% spacecraft (6 position and 6 velocity). 
%
% Triple S/C formations are also possible for simulations. In this case,
% estimated state vector consists of 9 position and 9 velocity states.
% If S/C initial states are selected from the same region (e.g.
% EML2 Halo) then it is suggested to choose two different IC. Otherwise, 
% S/C would have the same initial states. Two different network topologies
% are modelled: mesh (distributed) or centralized (star) network
% topologies. S/C in the mesh topology have an extra crosslink. This would
% provide additional information to the filter. 
%
% Clock parameters (bias, drift, and aging) are modeled and they can be
% either neglected, estimated or considered (only bias). However, clock
% parameter estimation or consider approach can not be used in the triple
% S/C formation. 
%
% 3 Different navigation data types are considered: Satellite-to-satellite 
% tracking (SST) based range, range-rate and angle. If more than one 
% measurement type is requested in the analysis then the system processes 
% all the navigation data at the same time. 
%
% In this version of this file to keep it simple, any visibility problem is 
% not considered, since based-on the experience, only 1-2% of the 
% measurements are blocked by the Moon if orbits are around different 
% Lagrangian points. Initial S/C states are provided by the database 
% IC_CRTBP.m file. 
% 
% Change Y0_true variable if user defined IC is requested. This file uses 
% CRTBP.m for propagation and STM_CRTBP.m, G_CRTBP.m for state transition 
% matrix calculations. In addition, UDFactor.m for UD factorization.
%
% Adaptive estimation of process noise covariance matrix is also considered
% in this file. In this case (Adaptive EKF), the forgetting parameter
% (alpha) is used in which 0.4 is found optimal for this problem. User may
% change this value, between 0-1. If alpha = 1, then AEKF turns into EKF.
% In addition, Consider KF, or Schmidt KF is also modelled. In this case,
% clock bias is the consider parameter. 
% User defined process noise matrix can also be used in this script. In
% this case, change the predefined sigm = 1e-12. Measurement noise
% covariance matrix is defined by the known measurement errors, so that
% the navigation system knows exactly the measurement noise. User may
% add errors into the measurement noise covariance matrix (find the line: 
% measnoisescale). ODE settings can be changed via the options line. In 
% addition, ode113 is known for orbital applications. This file provides
% output of estimation errors (position, velocity), uncertainty (position,
% velocity) and their RSS values. 
%
%
% Inputs:
% TSC           Third S/C included or not, logical 1 or 0
% Topo          Topology for 3 S/C formations, 0 = centralized, 1 = mesh
% P0            Initial State Covariance matrix m and m/s diag(12by12)
% Yerr          Initial State Error m and m/s (1by12)
% SC1           First S/C initial states from the database (1-16)
% SC2           Second S/C initial states from the database (1-16)
% SC3           Third S/C initial states from the database (if selected)
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
% estcb             Estimated clock bias logical 1 or 0 
% estcd             Estimated clock drift logical 1 or 0
% estca             Estimated clock aging logical 1 or 0
% consbias          Consider clock bias logical 1 or 0
% clockbias         True clock bias in meter
% clockdrift        True clock drift in meter/sec
% clockaging,       True clock aging in meter/sec^2
% Pclock            Clock parameter covariance matrix 3 by 3
% AF                Adaptive filtering logical 1 or 0
% measnoisescale    Measurement noise scale parameter, 1 default
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
% Note: This script is intended to use with CRTBP_OD_detailed_main.m
%       

function [rsspos, rssvel, rssposunc, rssvelunc, ErrYfull, SigUncfull] = ...
    CRTBP_OD_detailed_func(TSC, Topo, P0, Yerr, SC1, SC2, SC3, ...
    rm, rrm, ram, simdur, simstep, sigma_range, sigma_range_rate, ...
    sigma_rel_angle, bias_range, bias_rangerate, bias_rel_angle, estcb, ...
    estcd, estca, consbias, clockbias, clockdrift, clockaging, Pclock, ...
    AF, measnoisescale)

global AuxParam

AuxParam.tripleSC       = TSC; %Triple S/C case: 1 = true
AuxParam.topology       = Topo; %Topology: 0 = centralized, 1 = mesh
AuxParam.rangemeas      = rm;
AuxParam.rangeratemeas  = rrm;
AuxParam.relanglemeas   = ram;
AuxParam.AdapFilter     = AF; %0=false, 1=true
AuxParam.Clockparam     = 0; %0=neglected, 1=estimated, 2=considered
AuxParam.Smoother       = 0; %0=false, 1=true
AuxParam.infowindow     = 1; %0=false, 1=true
AuxParam.estclockbias   = estcb; %clock bias estimation, 0=false, 1=true
AuxParam.estclockdrift  = estcd; %clock drift estimation, 0=false, 1=true
AuxParam.estclockaging  = estca; %clock aging estimation, 0=false, 1=true
AuxParam.considerbias   = consbias; %considered bias, 0=false, 1=true

if (AuxParam.estclockbias || AuxParam.estclockdrift ||...
        AuxParam.estclockaging == 1) && AuxParam.tripleSC == 1
    error(['Clock parameters are not allowed to be estimated ' ...
        'in the triple S/C formations.'])
elseif (AuxParam.estclockbias || AuxParam.estclockdrift ||...
        AuxParam.estclockaging == 1) && (AuxParam.rangeratemeas || ...
        AuxParam.relanglemeas == 1)
    error(['Clock parameters can only be estimated via SST range' ...
        'measurements.'])
elseif (AuxParam.estclockbias || AuxParam.estclockdrift ||...
        AuxParam.estclockaging == 1) && (AuxParam.considerbias == 1)
    error(['Clock bias can not be estimated and considered at ' ...
        'the same time.'])
elseif (AuxParam.tripleSC == 1) && (AuxParam.considerbias == 1)
    error(['Consider parameters are not allowed in ' ...
        'the triple S/C formations.'])
end

LU=384747963;   %m, lenght unit conversion
SU=1.025351467559547e+03; %m/s, velocity convertion
ACU=0.0027; %m/s^2, acceleration conversion

sigma_range = (1/LU)*sigma_range;  %range measurement err in LU
sigma_range_rate = (1/SU)*sigma_range_rate; % range-rate meas err in SU
sigma_rel_angle = sigma_rel_angle;

% Measurement noise covariance matrix sigma values (no error is assumed)
sigma_range_R = measnoisescale*sigma_range;
sigma_range_rate_R = measnoisescale*sigma_range_rate;
sigma_rel_angle_R = measnoisescale*sigma_rel_angle;

% Measurement biases
bias_range=(1/LU)*bias_range;
bias_rangerate=(1/SU)*bias_rangerate;
bias_rel_angle=bias_rel_angle;

% Clock parameters
clockbias = (1/LU)*clockbias;       % Clock bias converted to non-dim LU
clockdrift = (1/SU)*clockdrift;     % Clock drift converted to non-dim SU
clockaging = (1/ACU)*clockaging;    % Clock aging converted to non-dim ACU

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

if AuxParam.tripleSC == 0   % double S/C system
    % Initial states
    Sc_1st=SC1;
    Sc_2nd=SC2;
    Y0_true = [IC(Sc_1st,:) IC(Sc_2nd,:)]';
    Yerr = Yerr.*[ones(1,3)*(1/LU) ones(1,3)*(1/SU) ones(1,3)*(1/LU) ...
        ones(1,3)*(1/SU)];

    % Expanding estimated state vector with clock parameters
    if AuxParam.estclockbias == 1   
        Y0_true = [Y0_true; clockbias];
        Yerr = [Yerr sqrt(Pclock(1))*(1/LU)];
    end
    if AuxParam.estclockdrift == 1
        Y0_true = [Y0_true; clockdrift];
        Yerr = [Yerr sqrt(Pclock(2))*(1/SU)];
    end
    if AuxParam.estclockaging == 1
        Y0_true = [Y0_true; clockaging];
        Yerr = [Yerr sqrt(Pclock(3))*(1/ACU)];
    end

    Y = Y0_true+Yerr';
    
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

    % Initial Process Covariance
    sigm = 1e-12;
    delt = step;
    Qdt1=[((delt^4)*sigm)/3 0 0 ((delt^3)*sigm)/2 0 0;
            0 ((delt^4)*sigm)/3 0 0 ((delt^3)*sigm)/2 0
            0 0 ((delt^4)*sigm)/3 0 0 ((delt^3)*sigm)/2
            ((delt^3)*sigm)/2 0 0 ((delt^2)*sigm) 0 0
            0 ((delt^3)*sigm)/2 0 0 ((delt^2)*sigm) 0
            0 0 ((delt^3)*sigm)/2 0 0 ((delt^2)*sigm)];
        
    Q=[Qdt1 zeros(6,6); zeros(6,6) Qdt1];
    sizematrix = 12;

    % Expanding state covariance matrix with clock parameters
    if AuxParam.estclockbias == 1   
        S = size(P);
        P = [P zeros(S(1),1); zeros(1,S(1)) Pclock(1)*((1/LU)^2)];
        Q = [Q zeros(S(1),1); zeros(1,S(1)) 0];
        sizematrix = sizematrix+1;
    end
    if AuxParam.estclockdrift == 1
        S = size(P);
        P = [P zeros(S(1),1); zeros(1,S(1)) Pclock(2)*((1/SU)^2)];
        Q = [Q zeros(S(1),1); zeros(1,S(1)) 0];
        sizematrix = sizematrix+1;
    end
    if AuxParam.estclockaging == 1
        S = size(P);
        P = [P zeros(S(1),1); zeros(1,S(1)) Pclock(3)*((1/ACU)^2)];
        Q = [Q zeros(S(1),1); zeros(1,S(1)) 0];
        sizematrix = sizematrix+1;
    end
    if AuxParam.considerbias == 1
        C = zeros(sizematrix,1);
    end
        

elseif AuxParam.tripleSC == 1 % triple S/C system
    sizematrix = 18;
    % Initial states
    Sc_1st=SC1;
    Sc_2nd=SC2;
    Sc_3rd=SC3;
    Y0_true = [IC(Sc_1st,:) IC(Sc_2nd,:) IC(Sc_3rd,:)]';
    Yerr = Yerr.*[ones(1,3)*(1/LU) ones(1,3)*(1/SU) ones(1,3)*(1/LU) ...
        ones(1,3)*(1/SU) ones(1,3)*(1/LU) ones(1,3)*(1/SU)];
    Y = Y0_true+Yerr';

    %Initial covariance matrix
    P = zeros(sizematrix);
    for i= 1:3
        P(i,i) = P0(i)*((1/LU)^2);
        P(i+6,i+6) = P0(i)*((1/LU)^2);
        P(i+12,i+12) = P0(i)*((1/LU)^2);
    end
    for i=4:6
        P(i,i) = P0(i)*((1/SU)^2);
        P(i+6,i+6) = P0(i)*((1/SU)^2);
        P(i+12,i+12) = P0(i)*((1/SU)^2);
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
        
    Q=[Qdt1 zeros(6,12); zeros(6,6) Qdt1 zeros(6,6); zeros(6,12) Qdt1];
    
end

%Gravational constant
G=1;
mu=0.012155650403207;

% Initialization
t = 0;

%Adaptive parameter (forgetting parameter)
if AuxParam.AdapFilter == 1
    alpha=0.4;
else
    alpha=1;
end

% Consider parameters
N=1;
B=clockbias*clockbias;
b=clockbias;

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

    % STATE PROPAGATION

    % Propagation to measurement epoch
    t = i*step;     % Time since epoch

    %Estimated trajectory
    % 1st S/C
    [~,xx]=ode113('CRTBP',[t_old t], Y(1:6)',options,[],1,mu);
    Y1=xx(end,:); 
    % 2nd S/C
    [~,xxx]=ode113('CRTBP',[t_old t], Y(7:12)',options,[],1,mu);
    Y2=xxx(end,:);

    if AuxParam.tripleSC == 1
        [~,xxx]=ode113('CRTBP',[t_old t], Y(13:18)',options,[],1,mu);
        Y3=xxx(end,:);
        % Combined States
        Y=[Y1 Y2 Y3]';
    else
        % Combined States
        % Y=[Y1 Y2]';
        Y(1:6)=Y1;
        Y(7:12)=Y2;
    end

    %True trajectory
    % 1st S/C
    [~,xx]=ode113('CRTBP',[t_old t], Y0_true(1:6)',options,[],1,mu);
    Y1=xx(end,:); 
    % 2nd S/C
    [~,xxx]=ode113('CRTBP',[t_old t], Y0_true(7:12)',options,[],1,mu);
    Y2=xxx(end,:);

    if AuxParam.tripleSC == 1
        [~,xxx]=ode113('CRTBP',[t_old t], Y0_true(13:18)',options,[],1,mu);
        Y3=xxx(end,:);
        % Combined States
        Y0_true=[Y1 Y2 Y3]';
    else
        % Combined States
        % Y0_true=[Y1 Y2]';
        Y0_true(1:6)=Y1;
        Y0_true(7:12)=Y2;
    end
    
    % State-Transition Matrix
    % 1st S/C
    STM=STM_CRTBP(t_old,t,Y_old(1:6),mu);
    % 2nd S/C
    STM1=STM_CRTBP(t_old,t,Y_old(7:12),mu);

    if AuxParam.tripleSC == 1
        % 1st S/C
        STM2=STM_CRTBP(t_old,t,Y_old(13:18),mu);
        % Combined STM
        Phi=[STM, zeros(6,12); zeros(6,6), STM1, zeros(6,6); ...
            zeros(6,12), STM2];
    else
        % Combined STM
        Phi=[STM, zeros(6,6); zeros(6,6), STM1];

        % Adding clock related partials to STM
        if AuxParam.estclockbias == 1   
            S = size(Phi);
            Phi = [Phi zeros(S(1),1); zeros(1,S(1)) 1];
        end
        if AuxParam.estclockdrift == 1
            S = size(Phi);
            Phi = [Phi zeros(S(1),1); zeros(1,S(1)) 1];
        end
        if AuxParam.estclockaging == 1
            S = size(Phi);
            Phi = [Phi zeros(S(1),1); zeros(1,S(1)) 1];
        end
    end

    %TIME UPDATE
    
    % Covariance Propagation
    P = Phi*P*Phi'+Q;

    % Consider covariance propagation
    if AuxParam.considerbias == 1
        C = Phi*C;
    end

    % Measurements and their partials
    dr        = Y(1:3)-Y(7:9); 
    [udr rho] = unit(dr);
    %Range
    if AuxParam.rangemeas == 1
        %measurement
        obs_range = norm(Y0_true(1:3)-Y0_true(7:9))+...
            normrnd(bias_range,sigma_range)+clockbias+(t*clockdrift)+...
            (t*t*clockaging);
        % range from model
        rangeHat = norm(dr);
        if AuxParam.estclockbias == 1   
            rangeHat = rangeHat + Y(13);
        end
        if AuxParam.estclockdrift == 1
            rangeHat = rangeHat + (t*Y(14));
        end
        if AuxParam.estclockaging == 1
            rangeHat = rangeHat + (t*t*Y(15));
        end
        Hrr = shiftdim(udr,-1);
        Hrv = zeros(size(Hrr));
        %Range measurement partials
        Hr   = [Hrr Hrv]; 
        if AuxParam.tripleSC == 1
            %measurement
            obs_range_3rd = norm(Y0_true(1:3)-Y0_true(13:15))+...
            normrnd(bias_range,sigma_range);
            %range from model
            dr_3rd        = Y(1:3)-Y(13:15); 
            [udr rho] = unit(dr_3rd);
            rangeHat_3rd = norm(dr_3rd);
            Hrr_3rd = shiftdim(udr,-1);
            Hrv_3rd = zeros(size(Hrr_3rd));
            %Range measurement partials
            Hr_3rd   = [Hrr_3rd Hrv_3rd]; 
            if AuxParam.topology == 1   %Mesh
                %measurement
                obs_range_3rd_mesh = norm(Y0_true(7:9)-Y0_true(13:15))+...
                normrnd(bias_range,sigma_range);
                %range from model
                dr_3rd_mesh = Y(7:9)-Y(13:15); 
                [udr rho] = unit(dr_3rd_mesh);
                rangeHat_3rd_mesh = norm(dr_3rd_mesh);
                Hrr_3rd_mesh = shiftdim(udr,-1);
                Hrv_3rd_mesh = zeros(size(Hrr_3rd));
                %Range measurement partials
                Hr_3rd_mesh   = [Hrr_3rd_mesh Hrv_3rd_mesh]; 
            end
        end
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
        [udr rho] = unit(dr);
        rangeHatrr = norm(dr);
        rau=[Y(1)-Y(7) Y(2)-Y(8) Y(3)-Y(9)];
        raudot=[Y(4)-Y(10) Y(5)-Y(11) Y(6)-Y(12)];
        rrate=dot(raudot,rau)/rangeHatrr;
        dv        = Y(4:6)-Y(10:12); 
        Hvr = shiftdim(dv.*repmat(1./rho,3,1) - ...
            dr.*repmat(dot(dr,dv)./rho.^3,3,1),-1); 
        Hvv = shiftdim(udr,-1);
        %Range-rate measurement partials
        Hv   = [Hvr Hvv]; 
        if AuxParam.tripleSC == 1
            %measurement
            rau_true_3rd=Y0_true(1:3)-Y0_true(13:15);
            Dist_true_3rd=norm(rau_true_3rd);
            raudot_true_3rd=Y0_true(4:6)-Y0_true(16:18);
            obs_rrate_3rd=(dot(raudot_true_3rd,rau_true_3rd)/Dist_true_3rd)+...
                normrnd(bias_rangerate,sigma_range_rate);
            %range-rate from model
            dr        = Y(1:3)-Y(13:15); 
            [udr rho] = unit(dr);
            rangeHat1 = norm(dr);
            rau=[Y(1)-Y(13) Y(2)-Y(14) Y(3)-Y(15)];
            raudot=[Y(4)-Y(16) Y(5)-Y(17) Y(6)-Y(18)];
            rrate_3rd=dot(raudot,rau)/rangeHat1;
            dv        = Y(4:6)-Y(16:18); 
            Hvr_3rd = shiftdim(dv.*repmat(1./rho,3,1) - ...
                dr.*repmat(dot(dr,dv)./rho.^3,3,1),-1); 
            Hvv_3rd = shiftdim(udr,-1);
            %Range-rate measurement partials centralized
            Hv_3rd   = [Hvr_3rd Hvv_3rd]; 
            if AuxParam.topology == 1   %Mesh
                %measurement
                rau_true_3rd_mesh=Y0_true(7:9)-Y0_true(13:15);
                Dist_true_3rd_mesh=norm(rau_true_3rd_mesh);
                raudot_true_3rd_mesh=Y0_true(10:12)-Y0_true(16:18);
                obs_rrate_3rd_mesh=...
                    (dot(raudot_true_3rd_mesh,rau_true_3rd_mesh)/...
                    Dist_true_3rd_mesh)+ ...
                    normrnd(bias_rangerate,sigma_range_rate);
                %range-rate from model
                dr        = Y(7:9)-Y(13:15); 
                [udr rho] = unit(dr);
                rangeHat2 = norm(dr);
                rau=[Y(7)-Y(13) Y(8)-Y(14) Y(9)-Y(15)];
                raudot=[Y(10)-Y(16) Y(11)-Y(17) Y(12)-Y(18)];
                rrate_3rd_mesh=dot(raudot,rau)/rangeHat2;
                dv        = Y(10:12)-Y(16:18); 
                Hvr_3rd_mesh = shiftdim(dv.*repmat(1./rho,3,1) - ...
                    dr.*repmat(dot(dr,dv)./rho.^3,3,1),-1); 
                Hvv_3rd_mesh = shiftdim(udr,-1);
                %Range-rate measurement partials mesh
                Hv_3rd_mesh   = [Hvr_3rd_mesh Hvv_3rd_mesh]; 
            end
        end
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
        rau=Y(1:3)-Y(7:9);
        Dist=norm(rau);
        az = atan2(Y(2)-Y(8),Y(1)-Y(7));
        el = atan2(Y(3)-Y(9),Dist);

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
        if AuxParam.tripleSC == 1
            %measurements
            rau_true_3rd=Y0_true(1:3)-Y0_true(13:15);
            Dist_true_3rd=norm(rau_true_3rd);
            obs_az_3rd = ...
                atan2(Y0_true(2)-Y0_true(14),Y0_true(1)-Y0_true(13))+...
                normrnd(bias_rel_angle,sigma_rel_angle);
            obs_el_3rd = atan2(Y0_true(3)-Y0_true(15),Dist_true_3rd)+...
                normrnd(bias_rel_angle,sigma_rel_angle);
    
            %angle from model
            rau=Y(1:3)-Y(13:15);
            Dist=norm(rau);
            az_3rd = atan2(Y(2)-Y(14),Y(1)-Y(13));
            el_3rd = atan2(Y(3)-Y(15),Dist);

            dr = Y(1:3) - Y(13:15);
            rho = norm(dr);
            denum = (rho.^3)*sqrt(1-(dr(3).^2 / rho.^2));
            Har(1,1) = dr(2)/(rho.^2);
            Har(1,2) = -dr(1)/(rho.^2);
            Har(2,1) = dr(1)*dr(3)/denum;
            Har(2,2) = dr(2)*dr(3)/denum;
            Har(2,3) = (dr(3).^2-rho.^2)/denum;
            Har(1,3:6) = zeros;
            Har(2,4:6) = zeros;
            Hang3_3rd = [Har -Har];
            if AuxParam.topology == 1
                %measurements
                rau_true_3rd_mesh=Y0_true(7:9)-Y0_true(13:15);
                Dist_true_3rd_mesh=norm(rau_true_3rd_mesh);
                obs_az_3rd_mesh = ...
                    atan2(Y0_true(8)-Y0_true(14),Y0_true(7)-Y0_true(13))+...
                    normrnd(bias_rel_angle,sigma_rel_angle);
                obs_el_3rd_mesh = ...
                    atan2(Y0_true(9)-Y0_true(15),Dist_true_3rd_mesh)+...
                    normrnd(bias_rel_angle,sigma_rel_angle);
        
                %angle from model
                rau=Y(7:9)-Y(13:15);
                Dist=norm(rau);
                az_3rd_mesh = atan2(Y(8)-Y(14),Y(7)-Y(13));
                el_3rd_mesh = atan2(Y(9)-Y(15),Dist);

                dr = Y(7:9) - Y(13:15);
                rho = norm(dr);
                denum = (rho.^3)*sqrt(1-(dr(3).^2 / rho.^2));
                Har(1,1) = dr(2)/(rho.^2);
                Har(1,2) = -dr(1)/(rho.^2);
                Har(2,1) = dr(1)*dr(3)/denum;
                Har(2,2) = dr(2)*dr(3)/denum;
                Har(2,3) = (dr(3).^2-rho.^2)/denum;
                Har(1,3:6) = zeros;
                Har(2,4:6) = zeros;
                Hang3_3rd_mesh = [Har -Har];
            end
        end
    end

    % Measurement partials and measurement covariance matrices
    H=[];
    R=[];
    d=[];
    if AuxParam.rangemeas == 1
        d=(obs_range - rangeHat);
        H=Hr;
        H=[H -H];
        if AuxParam.estclockbias == 1   
            H = [H 1];
        end
        if AuxParam.estclockdrift == 1
            H = [H t];
        end
        if AuxParam.estclockaging == 1
            H = [H t*t];
        end
        R=[sigma_range_R.^2];
        if AuxParam.tripleSC == 1
            d=[d; (obs_range_3rd - rangeHat_3rd)];
            H=[H zeros(1,6); Hr_3rd zeros(1,6) -Hr_3rd];
            R=[R sigma_range_R.^2];
            if AuxParam.topology == 1 %Mesh
                d=[d; (obs_range_3rd_mesh - rangeHat_3rd_mesh)];
                H=[H; zeros(1,6) Hr_3rd_mesh -Hr_3rd_mesh];
                R=[R sigma_range_R.^2];
            end
        end

    end
    if AuxParam.rangeratemeas  == 1
        d=[d;(obs_rrate - rrate)];
        if AuxParam.tripleSC == 0
            H=[H;Hv -Hv];
        elseif AuxParam.tripleSC == 1
            H=[H;Hv -Hv zeros(1,6)];
        end
        R=[R sigma_range_rate_R.^2];
        if AuxParam.tripleSC == 1
            d=[d;(obs_rrate_3rd - rrate_3rd)];
            H=[H;Hv_3rd zeros(1,6) -Hv_3rd];
            R=[R sigma_range_rate_R.^2];
            if AuxParam.topology == 1
                d=[d;(obs_rrate_3rd_mesh - rrate_3rd_mesh)];
                H=[H; zeros(1,6) Hv_3rd_mesh -Hv_3rd_mesh];
                R=[R sigma_range_rate_R.^2];
            end
        end
    end
    if AuxParam.relanglemeas == 1
        d=[d;az-obs_az; el-obs_el];
        if AuxParam.tripleSC == 0
            H=[H;Hang3];
        elseif AuxParam.tripleSC == 1
            H=[H;Hang3 zeros(2,6)];
        end
        R=[R sigma_rel_angle_R^2 sigma_rel_angle_R^2];
        if AuxParam.tripleSC == 1
            d=[d;az_3rd-obs_az_3rd; el_3rd-obs_el_3rd];
            H=[H; Hang3_3rd(1:2,1:6) zeros(2,6) Hang3_3rd(1:2,7:12)];
            R=[R sigma_rel_angle_R^2 sigma_rel_angle_R^2];
            if AuxParam.topology == 1
                d=[d;az_3rd_mesh-obs_az_3rd_mesh; el_3rd_mesh-obs_el_3rd_mesh];
                H=[H; zeros(2,6) Hang3_3rd_mesh];
                R=[R sigma_rel_angle_R^2 sigma_rel_angle_R^2];
            end
        end
    end
    R = diag(R);

    % MEASUREMENT UPDATE

    % Kalman gain
    if AuxParam.considerbias == 1
        % Lambda (check Frontiers paper for details)
        lambda=(H*P*H')+(N*C'*H')+(H*C*N')+(N*B*N')+R;
        
        % Kalman gain via UD factorization
        [U1,D1] = UDFactor(lambda,true);
        RHS=(P*H'+C*N')';
        X1=mldivide(U1,RHS);
        X2=mldivide(D1,X1);
        XF=mldivide(U1',X2);
        K=XF';
    else
        K = P*H'*inv(R+H*P*H');
    end

    % Correction
    if AuxParam.considerbias == 1
        d = d - N*b;
    end
    Y = Y + K*d;

    % Covariance update
    P = (eye(sizematrix)-K*H)*P;

    if AuxParam.considerbias == 1
         P = P - K*N*C';
         % Consider covariance update
         C = C - K*(H*C+N*B);
    end

    % Update Q
    Q = alpha*Q + (1-alpha)*(K*d*d'*K');

    % Estimation Errors
    ErrY = Y-Y0_true;
    Sigma = zeros(sizematrix,1);
    
    for ii=1:sizematrix
        Sigma(ii) = sqrt(P(ii,ii));
    end

    %RSS Pos and Vel errors
    if AuxParam.tripleSC == 1
        rsspos(k,:)=rssq([ErrY(1:3); ErrY(7:9); ErrY(13:15)]);
        rssvel(k,:)=rssq([ErrY(4:6); ErrY(10:12); ErrY(16:18)]);
        rssposunc(k,:)=rssq([Sigma(1:3); Sigma(7:9); Sigma(13:15)]);
        rssvelunc(k,:)=rssq([Sigma(4:6); Sigma(10:12); Sigma(16:18)]);
    else
        rsspos(k,:)=rssq([ErrY(1:3); ErrY(7:9)]);
        rssvel(k,:)=rssq([ErrY(4:6); ErrY(10:12)]);
        rssposunc(k,:)=rssq([Sigma(1:3); Sigma(7:9)]);
        rssvelunc(k,:)=rssq([Sigma(4:6); Sigma(10:12)]);
    end

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


