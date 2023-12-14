function y = Runge_Kutta_Merson(func,t,y0,dt, mu, n)
   %% Function Description
    % This function recieves a function to be solved, the time to start 
    % integration, the [state vector; Io] (for @sTM function, and the time
    % step to iterate with.
    % It implements the integration process from Runge-Kutta, and the core
    % code was "borrowed" from M. Mahooti via the MATLAB forums.
   %% Ethan Geipel, March 2019
   %% Global Variables (unused at the moment)
    % global mu
   %% The Function
    % dt = time step, Io = STM integration bit. y0 = [state vector; STM thing];
    % n = 1.99120675680384e-07; % n for Moon (?)
    % n = 0.0243736546460684; % n for Earth (?)
    % n=1; % For normalized case
    
    % Implementation of the Runge-Kutta integration approach
    eta0 = y0;
    k0   = dt*func(t,eta0, mu)*n;
    eta1 = eta0 + k0/3;
    k1   = dt*func(t+dt/3,eta1, mu)*n;
    eta2 = eta0 + (k0+k1)/6;
    k2   = dt*func(t+dt/3,eta2, mu)*n;
    eta3 = eta0 + (k0+3*k2)/8;
    k3   = dt*func(t+dt/2,eta3, mu)*n;
    eta4 = eta0 + (k0-3*k2+4*k3)/2;
    k4   = dt*func(t+dt,eta4, mu)*n;

    % Final return value
    y = eta0 + (k0+4*k3+k4)/6;

end