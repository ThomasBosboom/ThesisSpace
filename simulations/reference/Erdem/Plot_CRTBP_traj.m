%--------------------------------------------------------------------------
%
%                       Plot CRTBP Trajectory
%
%--------------------------------------------------------------------------
%% 
% (Last modified 22/Feb/2023)
% Inputs:
% Y     Satellite State vectors
% SC1   First S/C orbit info from the database
% SC2   Second S/C orbit info from the database
% 
% Output:
% 3D trajectory plot
%
% Example :
% ---------
% Satellite orbits: 
% 1:EML2south, 2:EML2south,3:EML2north, 4:EML2north 5:EML1south 6:EML1south
% 7:EML1north  8:EML1north 9:Lunar Elliptic 10:Lunar Polar-Circular
% 11:NRHO EML2south, 12:NRHO EML1south 13:NRHO EML2north, 
% 14:NRHO EML1north 15:EML2 Lyapunov 16:EML1 Lyapunov
%
% % Uncomment below and run
SC1=9;           % e.g. Lunar Elliptic (pick value from above orbits)
SC2=2;           % e.g. EML2south
simdur =50;      % simulation duration in days
step=0.001*86400;        % simulation step in seconds (gives states in each step)
Y=CRTBP_OD_traj(SC1, SC2, simdur, step);  % Run first trajectory file
%% 
% Orbital names, periods, etc.
IC_CRTBP

figure()
plot3(Y(:,1), Y(:,2), Y(:,3),'Linewidth',2)
text(Y(1,1)+0.005,Y(1,2),Y(1,3)+0.005,ICnames(SC1),...
    'interpreter','latex','FontSize', 12)
hold on;
plot3(Y(:,7), Y(:,8), Y(:,9),'Linewidth',2)
text(Y(1,7)+0.005,Y(1,8),Y(1,9)+0.005,ICnames(SC2),...
    'interpreter','latex','FontSize', 12)
hold on;
grid minor
xlabel('X [non-dim]','interpreter','latex','FontSize', 14);
ylabel('Y [non-dim]','interpreter','latex','FontSize', 14);
zlabel('Z [non-dim]','interpreter','latex','FontSize', 14);

% % Add Moon
% mu = 0.012155650403207;
% Rsy= 0.004515683426763;
% [xx, yy, zz] = sphere(1000);
% surf(1-mu+Rsy*xx, Rsy*yy, Rsy*zz)
% r = 0.8; g = r; b = r;
% map = [r g b
%        0 0 0
%        r g b];
% colormap(map)
% caxis([-Rsy/100 Rsy/100])
% shading interp
% hold on;

% Plot Lagrangian points L1 and L2
% plot3(L2(1),L2(2),L2(3),'o', Color='black')
% text(L2(1)+0.005,L2(2),L2(3)+0.005,'L2','interpreter','latex','FontSize', 12)
% hold on;
% plot3(L1(1),L1(2),L1(3),'o', Color='black')
% text(L1(1)+0.005,L1(2),L1(3)+0.005,'L1','interpreter','latex','FontSize', 12)
% hold on;

% Plot view
view(10,15);

% Save txt file
time = [0:step/86400:simdur]';
state_history = [time, Y];
dlmwrite('C:/Users/thoma/OneDrive/Documenten/GitHub/ThesisSpace/simulations/reference/Halo_orbit_files/Erdem_original.txt', state_history, 'delimiter', '\t', 'precision', 10)

view(10,15);

