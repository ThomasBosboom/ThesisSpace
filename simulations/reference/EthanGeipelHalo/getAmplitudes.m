% Specify the path to your text file
file_path = 'C:\Users\thoma\OneDrive\Documenten\GitHub\ThesisSpace\simulations\reference\DataLUMIO\TextFiles\LUMIO_Halo_Cj3p09_states_J2000_Earth_centered.txt';

% Use importdata to read the file
file_data = importdata(file_path);

% Extract numeric data (if available)
numeric_data = file_data.data;

% Extract text data (if available)
text_data = file_data.textdata;

% Now, you can use 'numeric_data' and 'text_data' as needed
disp(numeric_data);
disp(text_data);

x = numeric_data(:, 3);
y = numeric_data(:, 4);
z = numeric_data(:, 5);

figure(1);
plot3(x, y, z);
xlabel('X-axis');
ylabel('Y-axis');
zlabel('Z-axis');
title('3D Trajectory');


time_interval_between_points = 0.001
% Assuming 'x' represents the oscillatory motion
Fs = 1 / (time_interval_between_points); % Sampling frequency
N = length(z);
Y = fft(z);
P2 = abs(Y/N);
P1 = P2(1:N/2+1);
P1(2:end-1) = 2 * P1(2:end-1);
f = Fs * (0:(N/2))/N;

% Plot the amplitude spectrum
figure;
plot(f, P1);
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Frequency Spectrum');
