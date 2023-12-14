%--------------------------------------------------------------------------
%
%               CRTBP LiAISON application 
%               
%               Trajectory function - Single S/C
%
%--------------------------------------------------------------------------
%% 
% Inputs:
% SC        S/C orbit info from the database (between 1-16)
% simdur    simulation duration in days
% step      simulation step in seconds (gives states in each step) 
% 
% Output:
% Y     Satellite State vectors
%
% Note: It is possible to use different initial conditions than the ones
% from database. In that case, edit the line Y0_true. 
% (Last modified 22/Feb/2023)
%% 

function [Y] = CRTBP_OD_traj_single(SC1, simdur, step)

%load initial states
IC_CRTBP

%simulation duration in days
daysnum=simdur;
steptu = 2.6667e-06*step;

%9.752753388846403e-07 = 1mm/s 
%2.599104078947392e-06 = 1km
%384747.963km = 1LU
%1/4.343day   = 1TU
%1.025351467559547e+03 conversion parameter (LU/TU to m/s)

% 1:EML2south, 2:EML2south,3:EML2north, 4:EML2north 5:EML1south 6:EML1south,
% 7:EML1north  8:EML1north 9:Lunar Elliptic 10:Lunar Polar-Circular
% 11:NRHO EML2south, 12:NRHO EML1south 13:NRHO EML2north, 
% 14:NRHO EML1north 15:EML2 Lyapunov 16:EML1 Lyapunov

Sc_1st=SC1;

% Initial States
Y0_true = [IC(Sc_1st,:)];

%Gravational constant
G=1;
mu=0.012155650403207;

% Simulation duration in TU
t_end = (daysnum/4.343);

%Initial state 
Y = Y0_true;

% ODE option
options=odeset('RelTol',1e-12,'AbsTol',1e-22);

% Trajectory
[~,xx]=ode113('CRTBP',[0:steptu:t_end], Y(1:6)',options,[],1,mu);
Y=xx; 

end

