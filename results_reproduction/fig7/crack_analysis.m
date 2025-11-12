%% Crack Path Analysis in Porous Domains
% Analyze tortuosity and deviation from centerline as a function of porosity

clear; clc; close all;

% File names
files = {
    'ML_0_hole2.txt',
    'ML_10_hole2.txt',
    'ML_20_hole2.txt',
    'ML_30_hole2.txt',
    'ML_40_hole2.txt',
    'ML_50_hole2.txt'
};

porosity = [0, 10, 20, 30, 40, 50]; % Porosity levels in %

% Initialize storage
tortuosity = zeros(1, length(files));
deviation = zeros(1, length(files));

% Create figure for crack path visualization
figure('Position', [100 100 1400 800]);

%% Process each file
for i = 1:length(files)
    % Load data
    data = load(files{i});
    
    % Extract crack path (value = 1)
    [rows, cols] = find(data == 1);
    
    if isempty(rows)
        fprintf('Warning: No crack path found in file %d\n', i);
        continue;
    end
    
    % Sort crack path points from left to right
    [cols_sorted, sort_idx] = sort(cols);
    rows_sorted = rows(sort_idx);
    
    % Calculate actual crack path length (sum of distances between consecutive points)
    actual_length = 0;
    for j = 1:length(cols_sorted)-1
        dx = cols_sorted(j+1) - cols_sorted(j);
        dy = rows_sorted(j+1) - rows_sorted(j);
        actual_length = actual_length + sqrt(dx^2 + dy^2);
    end
    
    % Calculate straight-line distance (from initiation to termination)
    straight_line_distance = sqrt((cols_sorted(end) - cols_sorted(1))^2 + ...
                                  (rows_sorted(end) - rows_sorted(1))^2);
    
    % Calculate tortuosity
    if straight_line_distance > 0
        tortuosity(i) = actual_length / straight_line_distance;
    else
        tortuosity(i) = 1;
    end
    
    % Calculate deviation from centerline (horizontal reference)
    % Use the mean row position as the reference line
    mean_row = mean(rows_sorted);
    
    % Calculate perpendicular distances from each point to the horizontal reference
    perpendicular_distances = abs(rows_sorted - mean_row);
    
    % Mean deviation from centerline
    deviation(i) = mean(perpendicular_distances);
    
    % Visualize crack path
    subplot(2, 3, i);
    imagesc(data);
    colormap([1 1 1; 0 0 1; 1 0 0]); % white=intact, blue=pore, red=crack
    axis equal tight;
    title(sprintf('Porosity = %d%%\nTortuosity = %.3f\nDeviation = %.2f pixels', ...
                  porosity(i), tortuosity(i), deviation(i)));
    xlabel('X (pixels)');
    ylabel('Y (pixels)');
    
    % Add reference line
    hold on;
    plot([cols_sorted(1), cols_sorted(end)], [mean_row, mean_row], ...
         'g--', 'LineWidth', 2, 'DisplayName', 'Reference Line');
    hold off;
    
    % Print results
    fprintf('Porosity %d%%:\n', porosity(i));
    fprintf('  Tortuosity: %.4f\n', tortuosity(i));
    fprintf('  Mean Deviation: %.2f pixels\n', deviation(i));
    fprintf('  Actual path length: %.2f pixels\n', actual_length);
    fprintf('  Straight-line distance: %.2f pixels\n\n', straight_line_distance);
end

sgtitle('Crack Path Evolution with Porosity', 'FontSize', 14, 'FontWeight', 'bold');

%% Plot tortuosity in separate figure
figure('Position', [150 150 600 500]);
plot(porosity/10, tortuosity, 'bo-', 'LineWidth', 2, 'MarkerSize', 10, ...
     'MarkerFaceColor', 'b');
grid on;
xlabel('Porosity (%)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('Tortuosity $\tau$', 'FontSize', 14, 'Interpreter', 'latex');
set(gca, 'FontSize', 14);
xlim([0 5]);

% Save tortuosity figure
saveas(gcf, 'tortuosity_vs_porosity.png');
saveas(gcf, 'tortuosity_vs_porosity', 'epsc');   % color EPS
fprintf('Tortuosity figure saved to tortuosity_vs_porosity.png\n');

%% Plot deviation in separate figure
figure('Position', [150 150 600 500]);
plot(porosity/10, deviation, 'ro-', 'LineWidth', 2, 'MarkerSize', 10, ...
     'MarkerFaceColor', 'r');
grid on;
xlabel('Porosity (%)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('Mean Deviation from Centerline $D$ (pixels)', 'FontSize', 14, 'Interpreter', 'latex');
set(gca, 'FontSize', 14);
xlim([0 5]);

% Save deviation figure
saveas(gcf, 'deviation_vs_porosity.png');
saveas(gcf, 'deviation_vs_porosity', 'epsc');   % color EPS
fprintf('Deviation figure saved to deviation_vs_porosity.png\n');

%% Summary statistics
fprintf('\n=== SUMMARY ===\n');
fprintf('Porosity vs Tortuosity:\n');
for i = 1:length(porosity)
    fprintf('  %2d%%: %.4f\n', porosity(i), tortuosity(i));
end
fprintf('\nPorosity vs Deviation:\n');
for i = 1:length(porosity)
    fprintf('  %2d%%: %.2f pixels\n', porosity(i), deviation(i));
end

% Calculate correlation
if length(porosity) > 2
    corr_tort = corrcoef(porosity, tortuosity);
    corr_dev = corrcoef(porosity, deviation);
    fprintf('\nCorrelation with porosity:\n');
    fprintf('  Tortuosity: %.3f\n', corr_tort(1,2));
    fprintf('  Deviation: %.3f\n', corr_dev(1,2));
end

%% Save results
results = table(porosity', tortuosity', deviation', ...
                'VariableNames', {'Porosity_Percent', 'Tortuosity', 'Deviation_pixels'});
writetable(results, 'crack_analysis_results.csv');
fprintf('\nResults saved to crack_analysis_results.csv\n');
