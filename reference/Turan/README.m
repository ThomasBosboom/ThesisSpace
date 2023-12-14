%--------------------------------------------------------------------------
%
%          CRTBP Autonomous Orbit Determination application  
%               
%                           README
%
%          By Erdem Turan (Last modified 01/Mar/2023)
%
%--------------------------------------------------------------------------
%
% There are 3 different main Matlab scripts for the purpose of orbit 
% determination analysis: CRTBP_OD_CRLB_main.m, CRTBP_OD_EKF_main.m, and 
% CRTBP_OD_detailed_main.m. Please use these files for simulations.
% _________________________________________________________________________
%
% CRTBP_OD_CRLB_main.m (calls CRTBP_OD_CRLB_func.m function)
% This function provides the Cramer-Rao Lower Bound (CRLB) analysis for
% sequential filters. CRLB has been calculated based on inverse of the
% Fisher Information Matrix (FIM). 3 Different navigation data types are
% considered: Satellite-to-satellite tracking (SST) based range, range-rate
% and angle. If more than one measurement type is requested in the analysis 
% then the system processes all the measurements types at the same time.
% In this version of this file to keep it simple, any visibility 
% problem is not considered, since based on simulations, only 1-2% of 
% the measurements are blocked by the Moon if orbits are around different 
% Lagrangian points. Initial S/C states are provided by the database 
% IC_CRTBP.m file. Change Y0_true variable if user defined IC is requested.
% This file uses CRTBP.m for propagation and STM_CRTBP.m, G_CRTBP.m for
% state transition matrix calculations. 
%
% _________________________________________________________________________
%
% CRTBP_OD_EKF_main.m (calls CRTBP_EKF_func.m function) 
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
% based-on simulations, only 1-2% of the measurements are blocked by the 
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
% _________________________________________________________________________
%
% CRTBP_OD_detailed_main.m
% This function provides the sequential filter(extended Kalman filter, EKF)
% solution for satellite-to-satellite tracking based autonomous orbit 
% determination for the Circular Restricted Three-body problem (cislunar).
% This is more detailed than CRTBP_OD_EKF_main.m (3 S/C case,clock est.etc)
% Estimated state vector consists of position and velocity states of both
% spacecraft (6 position and 6 velocity). 
% Triple S/C formations are also possible for simulations. In this case,
% estimated state vector consists of 9 position and 9 velocity states.
% If S/C initial states are selected from the same region (e.g.
% EML2 Halo) then it is suggested to choose two different IC. Otherwise, 
% S/C would have the same initial states. Two different network topologies
% are modelled: mesh (distributed) or centralized (star) network
% topologies. S/C in the mesh topology have an extra crosslink. This would
% provide additional information to the filter. 
% Clock parameters (bias, drift, and aging) are modeled and they can be
% either neglected, estimated or considered (only bias). However, clock
% parameter estimation or consider approach can not be used in the triple
% S/C formation. 
% 3 Different navigation data types are considered: Satellite-to-satellite 
% tracking (SST) based range, range-rate and angle. If more than one 
% measurement type is requested in the analysis then the system processes 
% all the navigation data at the same time. 
% In this version of this file to keep it simple, any visibility problem is 
% not considered, since based-on the experience, only 1-2% of the 
% measurements are blocked by the Moon if orbits are around different 
% Lagrangian points. Initial S/C states are provided by the database 
% IC_CRTBP.m file. 
% Change Y0_true variable if user defined IC is requested. This file uses 
% CRTBP.m for propagation and STM_CRTBP.m, G_CRTBP.m for state transition 
% matrix calculations. In addition, UDFactor.m for UD factorization.
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
% Other files
% CRTBP.m for state propagation, STM_CRTBP.m  and sysSolveCRTBP.m for state 
% transition matrix,G_CRTBP Gradient calculations for state transition 
% matrix, IC_CRTBP data base: initial conditions and periodic informations, 
% unit.m for unit vector calculations, Plot_CRTBP_traj.m for trajectory 
% plots, Plot_CRTBP_traj_single.m for trajectory plots (single S/C),
% Plot_RSS_EKF.m for plotting EKF simulation results.