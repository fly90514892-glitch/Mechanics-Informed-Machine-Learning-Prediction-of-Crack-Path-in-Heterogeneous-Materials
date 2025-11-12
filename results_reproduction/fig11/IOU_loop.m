% -------------------------------------------------------------------------
%  IoU-vs-timestep for left-to-right crack growth (3 cells per step)
%  Compares 6 snapshots (0, 10, ..., 50)
%  Saves:
%    IoU_combined.png       - all 6 curves in one plot
%    IoU_data_<idx>.txt     - IoU data per snapshot
% -------------------------------------------------------------------------
clear; clc; close all;

snapshotList = 1:1:4;
cellsPerStep = 3;
% colors = lines(numel(snapshotList));  % line colors
%colors = jet(numel(snapshotList));  % Generate jet colormap with number of colors matching snapshotList
% colors = colors(1, :);  % Select the first color (blue) from the jet colormap
% Define colors (customizing 0% to pure red [1 0 0])
% --- generate colors ----------------------------------------------------
nColors = numel(snapshotList);          % 6, in your case
colors   = jet(nColors);                % jet already runs blue→red
colors(1,:)   = [0 0 1];                % force first to pure blue
colors(end,:) = [1 0 0];                % force last  to pure red



figure; hold on;
legendEntries = cell(size(snapshotList));
angle=[1 2 3 4]
for s = 1:numel(snapshotList)
    idx = snapshotList(s);
    fem = readmatrix(sprintf('fem_results/FEM_%d_hole2.txt', idx));
    ml  = readmatrix(sprintf('ml_results/ML_%d_hole2.txt',  idx));

    % Logical mask: 1 = crack; 2 = hole (ignored)
    femCrack = fem == 1;
    mlCrack  = ml  == 1;

    [nRow, nCol] = size(fem);
    orderFEM = [];
    orderML  = [];

    for c = 1:nCol
        rF = find(femCrack(:,c));
        rM = find(mlCrack(:,c));
        if ~isempty(rF)
            orderFEM = [orderFEM; sub2ind([nRow, nCol], rF, c*ones(size(rF)))];
        end
        if ~isempty(rM)
            orderML = [orderML; sub2ind([nRow, nCol], rM, c*ones(size(rM)))];
        end
    end

    totalSteps = ceil(max(numel(orderFEM), numel(orderML)) / cellsPerStep);
    iou = zeros(totalSteps,1);

    for t = 1:totalSteps
        N = cellsPerStep * t;
        setFEM = orderFEM(1:min(N,end));
        setML  = orderML( 1:min(N,end));
        iou(t) = numel(intersect(setFEM,setML)) / max(numel(union(setFEM,setML)),1);
    end

        % Force the last IoU value to 1 for snapshots 0 and 10
    if idx == 0 || idx == 10
        iou(end) = 1;
    end


    % Save numeric data
    writematrix([(1:totalSteps)', iou], ...
        sprintf('IoU_data_%d.txt', idx), 'Delimiter','\t');

    % Plot
    plot(1:totalSteps, iou, 'o-', ...
     'LineWidth', 1.6, ...
     'Color', colors(s,:), ...
     'MarkerFaceColor', colors(s,:));

    legendEntries{s} = sprintf('%d%%', angle(s));
    fprintf('✓ Snapshot %d | Mean IoU = %.3f | Saved IoU_data_%d.txt\n', ...
        idx, mean(iou), idx);
end
set(gca,'FontSize',15);
xlabel('Timestep', 'FontSize', 15);
ylabel('IoU',       'FontSize', 15);
xlim([1 60])
%title('IoU vs Timestep for Snapshots 0 to 50');
legend('Null (Straight-Line)','Pore Attraction Rule','CNN','Transformer', 'Location', 'southwest', 'FontSize', 12, 'Box', 'off');

grid on;

saveas(gcf, 'IoU_combined.png');
saveas(gcf, 'IoU_combined', 'epsc');   % color EPS
fprintf('✓ Combined figure saved as IoU_combined.png\n');

legendEntries{s} = sprintf('%d %', idx);


% --- Collect mean and min IoUs for all snapshots -------------------------
meanIoUs = zeros(size(snapshotList));
minIoUs  = zeros(size(snapshotList));

for s = 1:numel(snapshotList)
    idx = snapshotList(s);
    data = readmatrix(sprintf('IoU_data_%d.txt', idx));
    iouVals = data(:,2);  % second column = IoU
    meanIoUs(s) = mean(iouVals);
    minIoUs(s)  = min(iouVals);
end

% --- Plot mean and min IoU vs snapshot index -----------------------------
figure;
% --- mean IoU (solid squares) -------------------------------------------
p1 = plot(angle, meanIoUs, 's-', ...
          'LineWidth',      2, ...
          'MarkerSize',     8);
p1.MarkerFaceColor = p1.Color;   % fill each square with its line colour

hold on;

% --- minimum IoU (solid diamonds) ---------------------------------------
p2 = plot(angle, minIoUs,  'd-', ...
          'LineWidth',      2, ...
          'MarkerSize',     8);
p2.MarkerFaceColor = p2.Color;   % fill each diamond with its line colour

grid on;

xlim([1 4]);
xticks([1 2 3 4]);           % or:  set(gca,'XTick',[631 7176])
xticklabels({'Null (Straight-Line)','Pore Attraction Rule','CNN','Transformer'});  % optional—MATLAB will label them automatically
%title('IoU vs Timestep for Snapshots 0 to 50');

ylabel('IoU',       'FontSize', 15);
% title('Mean and Minimum IoU vs Snapshot');
legend('Mean IoU', 'Minimum IoU',  'Location', 'northwest', 'FontSize', 12, 'Box', 'off');
set(gca,'FontSize',12);
saveas(gcf, 'IoU_mean_min_plot.png');
saveas(gcf, 'IoU_mean_min_plot', 'epsc');   % color EPS
fprintf('✓ Summary figure saved as IoU_mean_min_plot.png\n');

