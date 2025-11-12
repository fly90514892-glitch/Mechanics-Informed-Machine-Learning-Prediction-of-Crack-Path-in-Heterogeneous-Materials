clc; clear all; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%parallal%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Retrieve the current cluster configuration
% c = parcluster('local');
% % Increase the maximum number of workers
% c.NumWorkers = 5; % Set it to the desired maximum number of workers
% % Save the modified cluster configuration
% saveProfile(c);
% % numberOfthread = 5;
% % mypool=parpool(numberOfthread);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%input%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
initialCase1 = 13;
finalCase1 = 14;
case_interval1 = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j = initialCase1:case_interval1:finalCase1

data_file2 = sprintf('map0_50.txt');
data_fem0 = readmatrix(data_file2, 'NumHeaderLines', 0);

data_file0 = sprintf('map%d.txt',j);
data_ml = readmatrix(data_file0, 'NumHeaderLines', 0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find the indices where matrix1 is equal to 1
indices = data_ml == 1;
data_ml(indices) = 2;
% Create a new combined matrix with the values from matrix1 where it's 1
combinedMatrix = data_ml;



% Find the indices where matrix1 is equal to 1
indices = data_fem0 == 1;
data_fem0(indices) = 1;
% Create a new combined matrix with the values from matrix1 where it's 1
combinedMatrix2 = data_fem0;

% Update the combined matrix with the values from matrix2 where matrix1 is not 1
combinedMatrix2(~indices) = combinedMatrix(~indices);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display the combined matrix
figure;
imagesc(combinedMatrix2);

% Define the colormap
colormap('jet');
caxis([0 3]);
% colorbar;

% Adjust the axes
axis equal tight;

% Set the labels
xlabel('$L_x$','Interpreter','latex','FontSize',20,'Color','k');
ylabel('$L_y$','Interpreter','latex','FontSize',20,'Color','k');

% Save the plot as an image
%set(gcf,'PaperUnits','inches','PaperPosition',[0 0 16 12]);
print('-dpng','-r1200',sprintf('plot_cpmpare%d.png',j));
print('-depsc','-r1200',sprintf('plot_cpmpare%d.eps',j));

close all;
end