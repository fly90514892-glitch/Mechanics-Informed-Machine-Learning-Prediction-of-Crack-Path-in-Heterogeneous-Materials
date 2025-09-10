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
connectivity_threshold = 5;
process_zone_size_x = 4;
process_zone_size_y = 9;

angle = zzz;

data_file = sprintf('map%d.txt',angle);
data1 = readmatrix(data_file, 'NumHeaderLines', 0);

% Check the crack path
% Initialize a data1 to store visited positions
visited = zeros(size(data1));

% Initialize variables to store the indices of connected elements
connected_indices = [];

% Loop through the data1 to find regions with connectivity_threshold or more connected elements with value 1
for i = 1:size(data1, 1)
    for j = 1:size(data1, 2)
        if data1(i, j) == 1 && visited(i, j) == 0
            % Use a depth-first search to count connected elements
            stack = [i, j];
            connected_count = 0;
            connected_positions = [];
            while ~isempty(stack)
                current = stack(1, :);
                stack(1, :) = [];
                x = current(1);
                y = current(2);
                if visited(x, y) == 0 && data1(x, y) == 1
                    visited(x, y) = 1;
                    connected_count = connected_count + 1;
                    connected_positions = [connected_positions; [x, y]];
                    % Define neighboring positions (up, down, left, right)
                    neighbors = [
                        x-1, y;
                        x+1, y;
                        x, y-1;
                        x, y+1;
                    ];
                    for k = 1:size(neighbors, 1)
                        nx = neighbors(k, 1);
                        ny = neighbors(k, 2);
                        if nx >= 1 && nx <= size(data1, 1) && ny >= 1 && ny <= size(data1, 2)
                            stack = [stack; nx, ny];
                        end
                    end
                end
            end
            % If the connected count meets the criterion, add to connected indices
            if connected_count >= connectivity_threshold
                connected_indices = [connected_indices; connected_positions];
            end
        end
    end
end
location = sortrows(connected_indices, 2);

% Check the crack tip (If the crack tip contains more than one element, select the middle one.)
% Initialize variables
row = size(location, 1);
column = size(location, 2);
previous_value = location(row, column);
rows_checked = 1;

% Start from the second last row and work upwards
for r = row - 1 : -1 : 1
    current_value = location(r, column);
    if current_value == previous_value
        rows_checked = rows_checked + 1;
    else
        break; % Exit the loop when they are not the same
    end
    previous_value = current_value;
end

% % Display the number of rows checked
% fprintf('Number of rows checked: %d\n', rows_checked);
numbers = 1:rows_checked;
crack_tip_row = row - round(median(numbers)) + 1;
crack_tip_locatrion = location(crack_tip_row,: );

%process_zone = zeros(process_zone_size, process_zone_size);
process_zone_row_start = crack_tip_locatrion(1) - (process_zone_size_y-1)/2;
process_zone_row_end = crack_tip_locatrion(1) + (process_zone_size_y-1)/2;
process_zone_column_start = crack_tip_locatrion(2);
process_zone_column_end = crack_tip_locatrion(2)+process_zone_size_x-1;

fid1 = fopen(sprintf('process_zone_location%d.txt',angle),'w');
fprintf(fid1,'%3d,%3d,%3d,%3d\n',...
   process_zone_row_start,...
   process_zone_row_end,...
   process_zone_column_start,...
   process_zone_column_end);   
fclose(fid1);
process_zone = data1(process_zone_row_start:process_zone_row_end, process_zone_column_start:process_zone_column_end);

process_zone_map = sprintf('process_zone%d.txt',angle);
writematrix(process_zone, process_zone_map, 'Delimiter', ',');

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
print('-dpng','-r600',sprintf('process_zone%d.png',angle));

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
print('-dpng','-r600',sprintf('map%d.png',angle));

fprintf('PROGRESS...case:%5d DONE \n',angle);    
close all;

