%--------------------------------------------------------------------------
%
%        CRTBP Orbit Determination main file
%
%        Cramer-Rao Lower Bound investigation for sequential filters
%
%        Main script to be able to run the CRTBP-CRLB file
%
%        By Erdem Turan (Last modified 22/Feb/2023)
%
%--------------------------------------------------------------------------
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
% Note: This script is used to run CRTBP_OD_CRLB_func.m
%       

clear all

% warning('off')

%% Simulation parameters

% Initial covariaance, sigma, meter, meter/s (diagonal elements)
P0 = [1000 1000 1000 1 1 1 1000 1000 1000 1 1 1];

% Conversion parameters
LU=384747963;   %m, lenght unit
SU=1.025351467559547e+03; %m/s, velocity convertion

% Initial states
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
SC1 = 1;

% S/C-2
SC2 = 2;

% Measurement types
rm = 1;          % Range measurement, logical either 1 or 0
rrm = 1;         % Range-rate measurement, logical either 1 or 0
ram = 0;         % Relative Angle measurement between S/C, az and el, ...
                 % logical either 1 or 0

simdur = 14;     % Simulation duration in days

% Meausrement cadance
simstep = 300;   % Simulation measurement step in seconds

% Measurement errors
sigma_range = 1;            % Range measurement error SD, m
sigma_range_rate = 1*1e-3;  % Range-rate measurement error SD, m/s
sigma_rel_angle = 20*(pi/648000);    % Azimuth and elevation angle ...
                                     % measurement error SD, arcsec

% Main CRLB function
[rssCRLB, rssCRLBvel, fullCRLB] = CRTBP_OD_CRLB_func(P0, SC1, SC2, ...
    rm, rrm, ram, simdur, simstep, sigma_range, sigma_range_rate, ...
    sigma_rel_angle);

disp('Simulation is done.')
cd = 7;   %Converged day (in general after 7 days for Libration points)

% Results
% position uncertainty
meanrssCRLB = mean(real(rssCRLB))*LU;
meanrssCRLBstable = mean(real(rssCRLB((3600/simstep)*24*cd:end)))*LU;
% velocity uncertainty
meanrssCRLB_vel = mean(real(rssCRLBvel))*SU;
meanrssCRLBstable_vel =mean(real(rssCRLBvel((3600/simstep)*24*cd:end)))*SU;

disp('-------------------------------------------------------------------')
disp(['mean RSS Position uncertainty (1sig)      : ' ...
    num2str(meanrssCRLB) ' m'])
disp(['mean RSS Position uncertainty (converged) : ' ...
    num2str(meanrssCRLBstable) ' m'])
disp(['mean RSS Velocity uncertainty (1sig)      : ' ...
    num2str(meanrssCRLB_vel*1000) ' mm/s'])
disp(['mean RSS Velocity uncertainty (converged) : ' ...
    num2str(meanrssCRLBstable_vel*1000) ' mm/s'])

%% Plot
% if full state results are required == 1
fullstate = 1;

% RSS uncertainty plots
Plot_RSS_CRTBP





