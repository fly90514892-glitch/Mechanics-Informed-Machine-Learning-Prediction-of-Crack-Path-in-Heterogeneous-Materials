clc; clear all; close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%parallal%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Retrieve the current cluster configuration
% c = parcluster('local');
% % Increase the maximum number of workers
% c.NumWorkers = 5; % Set it to the desired maximum number of workers
% % Save the modified cluster configuration
% saveProfile(c);
% numberOfthread = 5;
% mypool=parpool(numberOfthread);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%input%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
min_elements = 5;
process_zone_size_x = 4;
process_zone_size_y = 9;

angle = zzz;

data_file = sprintf('map%d.txt',angle);
data1 = readmatrix(data_file, 'NumHeaderLines', 0);

process_zone_location_file = sprintf('process_zone_location%d.txt',angle);
process_zone_location = readmatrix(process_zone_location_file, 'NumHeaderLines', 0);

process_zone_row_start = process_zone_location(1,1);
process_zone_row_end = process_zone_location(1,2);
process_zone_column_start = process_zone_location(1,3);
process_zone_column_end = process_zone_location(1,4);

process_zone_file = sprintf('process_zone_predict%d.txt',angle+1);
process_zone = readmatrix(process_zone_file, 'NumHeaderLines', 0);

data1(process_zone_row_start:process_zone_row_end, process_zone_column_start:process_zone_column_end) = process_zone(1:process_zone_size_y, 1:process_zone_size_x);

map_file = sprintf('map%d.txt',angle+1);
writematrix(data1, map_file,'Delimiter', ',');

% Plot the process zone with custom colors
figure;
imagesc(process_zone);

% Define the colormap
colormap('jet');
caxis([0 1]);
colorbar;

% Adjust the axes
axis equal tight;

% Set the labels
xlabel('$L_x$','Interpreter','latex','FontSize',20,'Color','k');
ylabel('$L_y$','Interpreter','latex','FontSize',20,'Color','k');

% Save the plot as an image
%set(gcf,'PaperUnits','inches','PaperPosition',[0 0 16 12]);
print('-dpng','-r600',sprintf('process_zone_predict%d.png',angle+1));

% Plot the domain with custom colors
figure;
imagesc(data1);

% Define the colormap
colormap('jet');
caxis([0 1]);
colorbar;

% Adjust the axes
axis equal tight;

% Set the labels
xlabel('$L_x$','Interpreter','latex','FontSize',20,'Color','k');
ylabel('$L_y$','Interpreter','latex','FontSize',20,'Color','k');

% Save the plot as an image
%set(gcf,'PaperUnits','inches','PaperPosition',[0 0 16 12]);
print('-dpng','-r600',sprintf('map%d.png',angle+1));

fprintf('PROGRESS...case:%5d DONE \n',angle);    
close all;



 
