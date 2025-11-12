clc;
clear all;
close all;

data_file1 = 'difference.txt';
data1 = readmatrix(data_file1, 'NumHeaderLines', 0);
accuracy = 100-(data1(:,1)/631)*100;

data_file2 = 'loss_data.txt';
data2 = readmatrix(data_file2, 'NumHeaderLines', 0);

barWidth = 1;

bar(data2(:, 1), [data2(:, 2), data2(:, 3)], barWidth, 'grouped');
ylim([0, 0.012]);
ylabel('Loss', 'Interpreter', 'latex', 'FontSize', 20, 'Color', 'k');
hold on;

yyaxis right
plot(data1(:, 2), accuracy, '-o', 'LineWidth', 3, 'Color',[0.9290 0.6940 0.1250]...
    , 'MarkerSize',5,'MarkerFaceColor',[0.9290 0.6940 0.1250]);
ax = gca;
ax.YColor = [0.9290 0.6940 0.1250]; % Set the color of the right y-axis to blue
hold off;
grid on;

% Set the labels
xlabel('Dense', 'Interpreter', 'latex', 'FontSize', 20, 'Color', 'k');
ylabel('Validation accuracy (\%)', 'Interpreter', 'latex', 'FontSize', 20, 'FontWeight','bold','Color', [0.9290 0.6940 0.1250]);
ylim([88, 100]);

% Add a legend
% Add a legend with box off and text size 10 at specific coordinates (x, y)
legend('training loss', 'validation loss', 'validation accuracy', 'Location', 'Best');
hLegend = legend('boxoff'); % Remove legend box
set(hLegend, 'FontSize', 12); % Change legend text size

% Set the legend position (adjust x and y coordinates accordingly)
legendX = 0.65; % x-coordinate
legendY = 0.75;  % y-coordinate
set(hLegend, 'Position', [legendX, legendY, 0.1, 0.1]); % [x, y, width, height]

% Save the plot as an image
print('-dpng', '-r1200', 'difference.png');
print('-depsc','-r1200','difference.eps');
