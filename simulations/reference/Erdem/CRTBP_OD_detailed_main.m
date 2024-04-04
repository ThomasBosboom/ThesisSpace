%--------------------------------------------------------------------------
%
%        CRTBP Autonomous Orbit Determination application 
%
%        MAIN FILE
%               
%        Detailed filter function (EKF, AEKF, AIEKF, CKF, etc.)
%        Triple S/C formation included (mesh or centralized topology)
%        Clock parameters included (neglected, estimated, considered)
%
%        Check function: CRTBP_OD_detailed_func.m for detailed explanation
%
%        By Erdem Turan (Last modified 23/Feb/2023)
%
%--------------------------------------------------------------------------
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
% Pclock            Clock parameter covariance matrix (3by3)
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
% Note: This script is used to run CRTBP_OD_detailed_func.m
% 

close all

% warning('off')

%% Simulation parameters

% Triple SC true =1, false =0;
TSC = 1;
% If Triple S/C scenario is considered, then:
% Topology: for Centralized = 0; Mesh =1;
Topo = 0;

% Initial covariaance, sigma, meter, meter/s (diagonal elements)
P0pos = 1000;
P0vel = 1;
P0 = [P0pos*ones(1,3) P0vel*ones(1,3)];
if TSC == 1
    P0 = [P0 P0 P0];
else
    P0 = [P0 P0];
end

% Clock related parameter covariance (diagonal elements)
% Bias (m)^2, drift (m/sec)^2, and aging (m/s^2)^2
Pclock = [100 0.01 0.001];

% Initial State Error
Yerr = sqrt(P0);

% Conversion parameters
LU=384747963;   %m, lenght unit
SU=1.025351467559547e+03; %m/s, velocity convertion
ACU=0.0027; %m/s^2, acceleration conversion

% Initial states from the database
%1:EML2south, 
%2:EML2south,
%3:EML2north, 
%4:EML2north 
%5:EML1south 
%6:EML1south,
%7:EML1north 
%8:EML1north
%9:Lunar Elliptic 
%10:Lunar Polar-Circular
%11:NRHO EML2south, 
%12:NRHO EML1south 
%13:NRHO EML2north, 
%14:NRHO EML1north 
%15:EML2 Lyapunov 
%16:EML1 Lyapunov

% S/C-1
SC1 = 2;

% S/C-2
SC2 = 9;

% S/C-3 (if triple SC system)
SC3 = 6;

% Measurement types (make it 1 in case )
rm = 1;          % Range measurement, logical either 1 or 0
rrm = 0;         % Range-rate measurement, logical either 1 or 0
ram = 0;         % Relative Angle measurement between S/C, az and el, logical either 1 or 0

simdur = 8;     % Simulation duration in days

% Meausrement cadance
simstep = 180;   % Simulation measurement step in seconds

% Measurement errors
sigma_range = 1;            % Range measurement error SD, m
sigma_range_rate = 1*1e-3;  % Range-rate measurement error SD, m/s
sigma_rel_angle = 20*(pi/648000);    % Azimuth and elevation angle measurement error SD, arcsec

% Filter info on measurement noise
measnoisescale = 1;    % Scale parameter e.g. measnoisescale*truevalue

% Measurement biases
bias_range=0;
bias_rangerate=0;
bias_rel_angle=0;

% Estimated clock parameters
estcb = 0;  %clock bias logical 1 or 0
estcd = 0;  %clock drift logical 1 or 0
estca = 0;   %clock aging logical 1 or 0

% Considered clock bias 
% If this is 1 then estimated parameters (estcb, estcd, estca) must be 0

consbias = 1;

% Set 0 if not estimated (neglected case otherwise)

clockbias =  10;     % True Clock bias in meters 
clockdrift = 0;  % True Clock drift in meters/sec
clockaging = 0;  % True Clock aging in meters/sec^2

% Adaptive Filtering
AF = 1;     % logical 1 or 0

% Main function
[rsspos, rssvel, rssposunc, rssvelunc, ErrYfull, SigUncfull] =...
    CRTBP_OD_detailed_func(TSC, Topo, P0, Yerr, SC1, SC2, SC3, rm, ...
    rrm, ram, simdur, simstep, sigma_range, sigma_range_rate, ...
    sigma_rel_angle, bias_range, bias_rangerate, bias_rel_angle, estcb, ...
    estcd, estca, consbias, clockbias, clockdrift, clockaging, Pclock, AF,...
    measnoisescale);

disp('Simulation is done.')
disp('-------------------------------------------------------------------')
disp(['mean RSS Position error              : ' num2str(mean(rsspos)*LU) ' m'])
disp(['mean RSS Position uncertainty (1sig) : ' num2str(mean(rssposunc)*LU) ' m'])
disp(['mean RSS Velocity error              : ' num2str(mean(rssvel)*1000*SU) ' mm/s'])
disp(['mean RSS Velocity uncertainty (1sig) : ' num2str(mean(rssvelunc)*1000*SU) ' mm/s'])

%% Plot
% if full state results are required = 1
fullstate = 1;

% RSS uncertainty plots
Plot_RSS_EKF

