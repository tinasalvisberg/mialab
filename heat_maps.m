close all;
clear all;

% Load the cleaned Excel file
filename = 'results_consistency_cleaned.xlsx';
data = readtable(filename, 'VariableNamingRule', 'preserve'); % Preserve original column names

% Extract unique brain labels and experiments
labels = unique(data.LABEL, 'stable'); % Maintain original order
experiments = data.Properties.VariableNames(4:end); % Extract experiment columns (from Baseline onwards)

% Initialize matrices for Dice and HD
diceMatrix = zeros(length(labels), length(experiments));
hdMatrix = zeros(length(labels), length(experiments));

% Fill the matrices with MEAN values for Dice and HDRFDST
for i = 1:length(labels)
    % Filter rows for each label and metric
    diceRow = data(strcmp(data.LABEL, labels{i}) & strcmp(data.METRIC, 'DICE') & strcmp(data.STATISTIC, 'MEAN'), 4:end);
    hdRow = data(strcmp(data.LABEL, labels{i}) & strcmp(data.METRIC, 'HDRFDST') & strcmp(data.STATISTIC, 'MEAN'), 4:end);
    
    % Convert table to array for plotting
    diceMatrix(i, :) = table2array(diceRow);
    hdMatrix(i, :) = table2array(hdRow);
end

% Plot the Dice heatmap
figure;
imagesc(diceMatrix);
colormap(flipud(turbo)); % Use 'hot' colormap reversed (dark for high values)
colorbar;
xticks(1:length(experiments));
xticklabels(experiments);
xlabel('Experiments');
yticks(1:length(labels));
yticklabels(labels);
ylabel('Brain Labels');
title('Dice Score Heatmap');

% Add numbers in the grid for Dice
for i = 1:length(labels)
    for j = 1:length(experiments)
        text(j, i, sprintf('%.3f', diceMatrix(i, j)), 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 10);
    end
end

% Plot the HD heatmap
figure;
imagesc(hdMatrix);
colormap(turbo); % Use 'hot' colormap (dark for low values)
colorbar;
xticks(1:length(experiments));
xticklabels(experiments);
xlabel('Experiments');
yticks(1:length(labels));
yticklabels(labels);
ylabel('Brain Labels');
title('Hausdorff Distance Heatmap');

% Add numbers in the grid for HD
for i = 1:length(labels)
    for j = 1:length(experiments)
        text(j, i, sprintf('%.2f', hdMatrix(i, j)), 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 10);
    end
end

%% Filter data excluding "White Matter" and "Grey Matter"
filteredData = data(~ismember(data.LABEL, {'WhiteMatter', 'GreyMatter'}), :);
filteredLabels = unique(filteredData.LABEL, 'stable');

% Initialize matrices for filtered Dice and HD
filteredDiceMatrix = zeros(length(filteredLabels), length(experiments));
filteredHdMatrix = zeros(length(filteredLabels), length(experiments));

% Fill matrices for filtered data
for i = 1:length(filteredLabels)
    diceRow = filteredData(strcmp(filteredData.LABEL, filteredLabels{i}) & strcmp(filteredData.METRIC, 'DICE') & strcmp(filteredData.STATISTIC, 'MEAN'), 4:end);
    hdRow = filteredData(strcmp(filteredData.LABEL, filteredLabels{i}) & strcmp(filteredData.METRIC, 'HDRFDST') & strcmp(filteredData.STATISTIC, 'MEAN'), 4:end);
    
    % Convert table to array for plotting
    filteredDiceMatrix(i, :) = table2array(diceRow);
    filteredHdMatrix(i, :) = table2array(hdRow);
end

% Plot filtered Dice heatmap
figure;
imagesc(filteredDiceMatrix);
colormap(flipud(turbo)); % Use 'hot' colormap reversed (dark for high values)
colorbar;
xticks(1:length(experiments));
xticklabels(experiments);
xlabel('Experiments');
yticks(1:length(filteredLabels));
yticklabels(filteredLabels);
ylabel('Brain Labels');
title('Filtered Dice Score Heatmap (Excluding White and Grey Matter)');

% Add numbers in the grid for filtered Dice
for i = 1:length(filteredLabels)
    for j = 1:length(experiments)
        text(j, i, sprintf('%.3f', filteredDiceMatrix(i, j)), 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 10);
    end
end

% Plot filtered HD heatmap
figure;
imagesc(filteredHdMatrix);
colormap(turbo); % Use 'hot' colormap (dark for low values)
colorbar;
xticks(1:length(experiments));
xticklabels(experiments);
xlabel('Experiments');
yticks(1:length(filteredLabels));
yticklabels(filteredLabels);
ylabel('Brain Labels');
title('Filtered Hausdorff Distance Heatmap (Excluding White and Grey Matter)');

% Add numbers in the grid for filtered HD
for i = 1:length(filteredLabels)
    for j = 1:length(experiments)
        text(j, i, sprintf('%.2f', filteredHdMatrix(i, j)), 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 10);
    end
end
