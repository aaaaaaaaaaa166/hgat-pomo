% Plot business-constraint stability comparisons for the 80/50 target.
% Run from the repository root or from this script's directory.

clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
outDir = fullfile(scriptDir, 'figures');

% If MATLAB runs a copied script from C:\Users\...\MATLAB, mfilename points
% there instead of the repo. Keep a manual override available, then search
% the common repo/script locations.
manualCsvPath = "";

candidateCsvPaths = [
    string(manualCsvPath)
    string(fullfile(scriptDir, 'reports', 'final_business_constraint_summary.csv'))
    string(fullfile(pwd, 'reports', 'final_business_constraint_summary.csv'))
    string(fullfile(scriptDir, '..', '..', ...
        'experiments', 'business_constraint_sensitivity_80_50_stability', ...
        'reports', 'final_business_constraint_summary.csv'))
    string(fullfile(pwd, ...
        'experiments', 'business_constraint_sensitivity_80_50_stability', ...
        'reports', 'final_business_constraint_summary.csv'))
    "D:\HGAT-POMO\experiments\business_constraint_sensitivity_80_50_stability\reports\final_business_constraint_summary.csv"
];

csvPath = "";
for p = reshape(candidateCsvPaths, 1, [])
    if strlength(p) > 0 && isfile(p)
        csvPath = p;
        break;
    end
end

if strlength(csvPath) == 0
    disp('Searched CSV paths:');
    disp(candidateCsvPaths);
    error(['Cannot find final_business_constraint_summary.csv. ' ...
           'Set manualCsvPath at the top of this script or cd to d:\HGAT-POMO first.']);
end

if strlength(scriptDir) == 0
    scriptDir = pwd;
end
if ~contains(string(scriptDir), "business_constraint_sensitivity_80_50_stability")
    outDir = fullfile(fileparts(char(csvPath)), '..', 'figures');
end
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

fprintf('Using summary CSV: %s\n', csvPath);
fprintf('Saving figures to: %s\n', outDir);

T = readtable(char(csvPath), 'TextType', 'string');
T.config_short = erase(T.experiment_name, "combined_");
T.reaches_80_50 = T.acceptance_rate >= 0.80 & ...
                  T.on_time_rate >= 0.50 & ...
                  T.hard_constraint_violations == 0;

configs = ["D", "E", "F", "G"];
evals = [50, 100];
oracleOnTime = T(T.method_name == "oracle_best_on_time", :);
oracleAccept = T(T.method_name == "oracle_best_acceptance", :);

%% Figure 1: recommended oracle_best_on_time stability across D/E/F/G
acc = nan(numel(configs), numel(evals));
ontime = nan(numel(configs), numel(evals));
for i = 1:numel(configs)
    for j = 1:numel(evals)
        row = oracleOnTime(oracleOnTime.config_short == configs(i) & oracleOnTime.eval_instances == evals(j), :);
        if height(row) > 0
            acc(i, j) = row.acceptance_rate(1);
            ontime(i, j) = row.on_time_rate(1);
        end
    end
end

figure('Color', 'w', 'Position', [100 100 980 420]);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
bar(acc, 'grouped');
yline(0.80, '--r', '80% target', 'LineWidth', 1.2);
ylim([0 1.05]);
set(gca, 'XTickLabel', configs);
xlabel('Configuration');
ylabel('Acceptance rate');
title('Acceptance: oracle\_best\_on\_time');
legend("eval=50", "eval=100", 'Location', 'southoutside', 'Orientation', 'horizontal');
grid on;

nexttile;
bar(ontime, 'grouped');
yline(0.50, '--r', '50% target', 'LineWidth', 1.2);
ylim([0 1.05]);
set(gca, 'XTickLabel', configs);
xlabel('Configuration');
ylabel('On-time rate');
title('On-time: oracle\_best\_on\_time');
legend("eval=50", "eval=100", 'Location', 'southoutside', 'Orientation', 'horizontal');
grid on;

saveas(gcf, fullfile(outDir, 'fig1_oracle_on_time_stability.png'));

%% Figure 2: method comparison at eval=100
methods = ["raw_baseline", "v2_repair_only", ...
           "tail_risk_constrained_joint_beam", ...
           "oracle_best_acceptance", "oracle_best_on_time"];
methodLabels = ["raw", "v2 repair", "tail-risk ref.", "oracle acc.", "oracle on-time"];

T100 = T(T.eval_instances == 100, :);
acc100 = nan(numel(configs), numel(methods));
ontime100 = nan(numel(configs), numel(methods));
for i = 1:numel(configs)
    for j = 1:numel(methods)
        row = T100(T100.config_short == configs(i) & T100.method_name == methods(j), :);
        if height(row) > 0
            acc100(i, j) = row.acceptance_rate(1);
            ontime100(i, j) = row.on_time_rate(1);
        end
    end
end

figure('Color', 'w', 'Position', [100 100 1080 460]);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
bar(acc100, 'grouped');
yline(0.80, '--r', '80% target', 'LineWidth', 1.2);
ylim([0 1.05]);
set(gca, 'XTickLabel', configs);
xlabel('Configuration');
ylabel('Acceptance rate');
title('Method comparison at eval=100');
legend(methodLabels, 'Location', 'southoutside', 'Orientation', 'horizontal');
grid on;

nexttile;
bar(ontime100, 'grouped');
yline(0.50, '--r', '50% target', 'LineWidth', 1.2);
ylim([0 1.05]);
set(gca, 'XTickLabel', configs);
xlabel('Configuration');
ylabel('On-time rate');
title('Method comparison at eval=100');
legend(methodLabels, 'Location', 'southoutside', 'Orientation', 'horizontal');
grid on;

saveas(gcf, fullfile(outDir, 'fig2_method_comparison_eval100.png'));

%% Figure 3: oracle trade-off at eval=100
oracle100 = T100(ismember(T100.method_name, ["oracle_best_acceptance", "oracle_best_on_time"]), :);
oracle100.label = oracle100.config_short + newline + replace(oracle100.method_name, "oracle_best_", "");

figure('Color', 'w', 'Position', [100 100 980 520]);
scatter(oracle100.acceptance_rate, oracle100.on_time_rate, 90, ...
        oracle100.average_lateness, 'filled');
xline(0.80, '--r', '80% acc.', 'LineWidth', 1.2);
yline(0.50, '--r', '50% on-time', 'LineWidth', 1.2);
text(oracle100.acceptance_rate + 0.004, oracle100.on_time_rate, oracle100.label, ...
     'FontSize', 9);
xlim([0.80 1.02]);
ylim([0.45 0.70]);
xlabel('Acceptance rate');
ylabel('On-time rate');
title('Oracle trade-off at eval=100 (color = avg. lateness)');
cb = colorbar;
cb.Label.String = 'Average lateness';
grid on;

saveas(gcf, fullfile(outDir, 'fig3_oracle_tradeoff_eval100.png'));

%% Figure 4: lateness and cost comparison for oracle_best_on_time
avgLate = nan(numel(configs), numel(evals));
maxLate = nan(numel(configs), numel(evals));
energy = nan(numel(configs), numel(evals));
distance = nan(numel(configs), numel(evals));
for i = 1:numel(configs)
    for j = 1:numel(evals)
        row = oracleOnTime(oracleOnTime.config_short == configs(i) & oracleOnTime.eval_instances == evals(j), :);
        if height(row) > 0
            avgLate(i, j) = row.average_lateness(1);
            maxLate(i, j) = row.max_lateness(1);
            energy(i, j) = row.total_energy_consumption(1);
            distance(i, j) = row.total_flight_distance(1);
        end
    end
end

figure('Color', 'w', 'Position', [100 100 1080 700]);
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
bar(avgLate, 'grouped');
set(gca, 'XTickLabel', configs);
ylabel('Avg. lateness');
title('Average lateness');
legend("eval=50", "eval=100", 'Location', 'best');
grid on;

nexttile;
bar(maxLate, 'grouped');
set(gca, 'XTickLabel', configs);
ylabel('Max lateness');
title('Maximum lateness');
legend("eval=50", "eval=100", 'Location', 'best');
grid on;

nexttile;
bar(energy, 'grouped');
set(gca, 'XTickLabel', configs);
ylabel('Energy');
title('Total energy consumption');
legend("eval=50", "eval=100", 'Location', 'best');
grid on;

nexttile;
bar(distance, 'grouped');
set(gca, 'XTickLabel', configs);
ylabel('Distance');
title('Total flight distance');
legend("eval=50", "eval=100", 'Location', 'best');
grid on;

saveas(gcf, fullfile(outDir, 'fig4_lateness_cost_oracle_on_time.png'));

%% Figure 5: trained ServicePolicy gate under default_business_env
candidateServiceCsvPaths = [
    string(fullfile(scriptDir, '..', 'default_business_env_training', ...
        'metrics', 'augmented_severe8_assignment_gate_summary.csv'))
    string(fullfile(pwd, ...
        'experiments', 'default_business_env_training', ...
        'metrics', 'augmented_severe8_assignment_gate_summary.csv'))
    "D:\HGAT-POMO\experiments\default_business_env_training\metrics\augmented_severe8_assignment_gate_summary.csv"
];

serviceCsvPath = "";
for p = reshape(candidateServiceCsvPaths, 1, [])
    if strlength(p) > 0 && isfile(p)
        serviceCsvPath = p;
        break;
    end
end

if strlength(serviceCsvPath) > 0
    fprintf('Using ServicePolicy gate CSV: %s\n', serviceCsvPath);
    S = readtable(char(serviceCsvPath), 'TextType', 'string');

    serviceEvals = [20, 50, 100];
    serviceMethods = ["raw_baseline", "oracle_best_on_time", "service_policy_imitation"];
    serviceLabels = ["raw", "oracle", "ServicePolicy"];

    serviceAcc = nan(numel(serviceEvals), numel(serviceMethods));
    serviceOnTime = nan(numel(serviceEvals), numel(serviceMethods));
    serviceAvgLate = nan(numel(serviceEvals), numel(serviceMethods));
    serviceMaxLate = nan(numel(serviceEvals), numel(serviceMethods));

    for i = 1:numel(serviceEvals)
        for j = 1:numel(serviceMethods)
            row = S(S.eval_instances == serviceEvals(i) & S.method == serviceMethods(j), :);
            if height(row) > 0
                serviceAcc(i, j) = row.acceptance_rate(1);
                serviceOnTime(i, j) = row.on_time_rate(1);
                serviceAvgLate(i, j) = row.average_lateness(1);
                serviceMaxLate(i, j) = row.max_lateness(1);
            end
        end
    end

    figure('Color', 'w', 'Position', [100 100 1080 460]);
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    bar(serviceAcc, 'grouped');
    yline(0.80, '--r', '80% target', 'LineWidth', 1.2);
    ylim([0 1.05]);
    set(gca, 'XTickLabel', string(serviceEvals));
    xlabel('Eval instances');
    ylabel('Acceptance rate');
    title('Default env acceptance');
    legend(serviceLabels, 'Location', 'southoutside', 'Orientation', 'horizontal');
    grid on;

    nexttile;
    bar(serviceOnTime, 'grouped');
    yline(0.50, '--r', '50% target', 'LineWidth', 1.2);
    ylim([0 1.05]);
    set(gca, 'XTickLabel', string(serviceEvals));
    xlabel('Eval instances');
    ylabel('On-time rate');
    title('Default env on-time');
    legend(serviceLabels, 'Location', 'southoutside', 'Orientation', 'horizontal');
    grid on;

    saveas(gcf, fullfile(outDir, 'fig5_service_policy_gate_rates.png'));

    %% Figure 6: trained ServicePolicy lateness profile
    figure('Color', 'w', 'Position', [100 100 1080 460]);
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile;
    bar(serviceAvgLate, 'grouped');
    set(gca, 'XTickLabel', string(serviceEvals));
    xlabel('Eval instances');
    ylabel('Avg. lateness');
    title('Default env average lateness');
    legend(serviceLabels, 'Location', 'southoutside', 'Orientation', 'horizontal');
    grid on;

    nexttile;
    bar(serviceMaxLate, 'grouped');
    set(gca, 'XTickLabel', string(serviceEvals));
    xlabel('Eval instances');
    ylabel('Max lateness');
    title('Default env maximum lateness');
    legend(serviceLabels, 'Location', 'southoutside', 'Orientation', 'horizontal');
    grid on;

    saveas(gcf, fullfile(outDir, 'fig6_service_policy_lateness.png'));
else
    warning('ServicePolicy gate CSV not found; skipping Figure 5/6.');
end

%% Console summary
disp('Figures saved to:');
disp(outDir);

disp('Rows reaching 80/50 with zero hard violations:');
disp(T(T.reaches_80_50, ...
    {'experiment_name', 'method_name', 'eval_instances', ...
     'acceptance_rate', 'on_time_rate', 'hard_constraint_violations'}));
