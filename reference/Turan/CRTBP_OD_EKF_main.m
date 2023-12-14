%--------------------------------------------------------------------------
%
%            CRTBP Orbit Determination main file
%
%            Estimation script (EKF)
%
%            Main script to run the CRTBP-EKF
%
%            By Erdem Turan (Last modified 22/Feb/2023)
%
%--------------------------------------------------------------------------
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
% Note: This script is used to run CRTBP_OD_EKF_func.m
% 

clear all

% warning('off')

%% Simulation parameters

% Initial covariaance, sigma, meter, meter/s (diagonal elements)
P0 = [1000 1000 1000 1 1 1 1000 1000 1000 1 1 1];

% Initial State Error
Yerr = sqrt(P0);
%Yerr = [100 100 100 0.1 0.1 0.1 100 100 100 0.1 0.1 0.1];

% Conversion parameters
LU=384747963;   %m, lenght unit
SU=1.025351467559547e+03; %m/s, velocity convertion

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
SC1 = 1;

% S/C-2
SC2 = 9;

% Measurement types (make it 1 in case )
rm = 1;          % Range measurement, logical either 1 or 0
rrm = 0;         % Range-rate measurement, logical either 1 or 0
ram = 0;         % Relative Angle measurement between S/C, az and el, logical either 1 or 0

simdur = 14;     % Simulation duration in days

% Meausrement cadance
simstep = 180;   % Simulation measurement step in seconds

% Measurement errors
sigma_range = 1;            % Range measurement error SD, m
sigma_range_rate = 1*1e-3;  % Range-rate measurement error SD, m/s
sigma_rel_angle = 20*(pi/648000);    % Azimuth and elevation angle measurement error SD, arcsec

% Measurement biases
bias_range=0;
bias_rangerate=0;
bias_rel_angle=0;

% Main function
[rsspos, rssvel, rssposunc, rssvelunc, ErrYfull, SigUncfull] = ...
    CRTBP_OD_EKF_func(P0, Yerr, SC1, SC2, rm, rrm, ram, simdur, simstep, ...
    sigma_range, sigma_range_rate, sigma_rel_angle, ...
    bias_range, bias_rangerate, bias_rel_angle);

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





