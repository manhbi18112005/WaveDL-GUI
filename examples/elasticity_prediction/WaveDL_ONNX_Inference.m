%% WaveDL ONNX Model Inference in MATLAB
% =========================================================================
% Demonstrates inference using a WaveDL-exported ONNX model in MATLAB.
% The model includes de-normalization, so outputs are in physical units.
%
% Requirements:
%   - MATLAB R2024a or later (for ONNX opset 17 support)
%   - Deep Learning Toolbox
%   - Deep Learning Toolbox Converter for ONNX Model Format
%
% Author: Ductho Le (ductho.le@outlook.com)
% =========================================================================

clear; clc;

%% ========================= CONFIGURATION ================================
ONNX_MODEL_PATH = 'model.onnx';
TEST_DATA_PATH  = 'Test_data_100.mat';
INPUT_SIZE      = [500, 500];  % Expected [H, W] for network initialization

%% ========================= LOAD MODEL ===================================
fprintf('📦 Loading ONNX model: %s\n', ONNX_MODEL_PATH);

% Import and initialize network (suppress all warnings during import)
warnState = warning('off', 'all');
net = importNetworkFromONNX(ONNX_MODEL_PATH);
warning(warnState);

exampleInput = dlarray(single(zeros([INPUT_SIZE, 1])), 'SSC');
net = initialize(net, exampleInput);

fprintf('   ✔ Model loaded (%d layers)\n', numel(net.Layers));

%% ========================= LOAD DATA ====================================
fprintf('\n📂 Loading test data: %s\n', TEST_DATA_PATH);
data = load(TEST_DATA_PATH);

X_raw = data.input_test;
Y     = data.output_test;

fprintf('   Raw shape: [%s]\n', join(string(size(X_raw)), ' × '));

% Preprocess: handle MATLAB/Python dimension ordering differences
X = preprocessData(X_raw, size(Y, 1));

fprintf('   ✔ Loaded %d samples\n', size(X, 1));

%% ========================= INFERENCE ====================================
fprintf('\n🚀 Running inference...\n');

N = size(X, 1);
numOutputs = size(Y, 2);
predictions = zeros(N, numOutputs, 'single');

tic;
for i = 1:N
    sample = prepareSample(X, i);
    dlSample = dlarray(sample, 'SSC');
    dlPred = predict(net, dlSample);
    predictions(i, :) = extractdata(dlPred)';
end
elapsedTime = toc;

fprintf('   ✔ Completed in %.2f s (%.1f ms/sample)\n', elapsedTime, elapsedTime/N*1000);

%% ========================= METRICS ======================================
fprintf('\n📊 Evaluation Metrics:\n');

errors = predictions - Y;
metrics = struct(...
    'R2',   computeR2(Y, predictions), ...
    'RMSE', sqrt(mean(errors.^2, 'all')), ...
    'MAE',  mean(abs(errors), 'all'));

fprintf('   R² Score: %.6f\n', metrics.R2);
fprintf('   RMSE:     %.6f\n', metrics.RMSE);
fprintf('   MAE:      %.6f\n', metrics.MAE);

% Per-parameter breakdown
fprintf('\n   Per-Parameter MAE:\n');
for p = 1:numOutputs
    fprintf('     P%d: %.6f\n', p, mean(abs(errors(:, p))));
end

%% ========================= VISUALIZATION ================================
fprintf('\n📈 Generating plots...\n');

fig = figure('Position', [100, 100, 400*numOutputs, 350], 'Color', 'w');

for p = 1:numOutputs
    ax = subplot(1, numOutputs, p);

    scatter(Y(:, p), predictions(:, p), 25, 'filled', ...
        'MarkerFaceAlpha', 0.6, 'MarkerFaceColor', [0.2 0.4 0.8]);
    hold on;

    % Perfect prediction line
    lims = [min([Y(:, p); predictions(:, p)]), ...
        max([Y(:, p); predictions(:, p)])];
    plot(lims, lims, 'r--', 'LineWidth', 1.5);

    % Styling
    xlabel(sprintf('True P%d', p), 'FontWeight', 'bold');
    ylabel(sprintf('Predicted P%d', p), 'FontWeight', 'bold');
    title(sprintf('Parameter %d (R² = %.4f)', p, computeR2(Y(:,p), predictions(:,p))));
    grid on; axis equal; axis tight;
    ax.FontSize = 10;
    hold off;
end

sgtitle('WaveDL ONNX Model - Predictions vs Ground Truth', 'FontWeight', 'bold');

% Disable interactive toolbar before export to avoid warning
set(fig, 'ToolBar', 'none');
exportgraphics(fig, 'onnx_results.png', 'Resolution', 300);

fprintf('   ✔ Saved: onnx_results.png\n');
fprintf('\n✅ Inference Complete\n');

%% ========================= HELPER FUNCTIONS =============================

function X = preprocessData(X_raw, numSamples)
% Handle dimension transposition from Python/MATLAB format differences
dims = size(X_raw);

if dims(end) == numSamples && numel(dims) >= 2
    % Data is transposed: (H, W, N) or (H, W, C, N)
    if numel(dims) == 3
        X_raw = permute(X_raw, [3, 1, 2]);
    elseif numel(dims) == 4
        X_raw = permute(X_raw, [4, 3, 1, 2]);
    end
end

% Ensure 4D: (N, C, H, W)
if ndims(X_raw) == 3
    X_raw = reshape(X_raw, [size(X_raw, 1), 1, size(X_raw, 2), size(X_raw, 3)]);
end

X = X_raw;
end

function sample = prepareSample(X, idx)
% Extract and format single sample for dlarray
sample = squeeze(X(idx, :, :, :));

if ismatrix(sample)
    % 2D: transpose for row-major ordering, add channel
    sample = single(sample');
    sample = reshape(sample, [size(sample, 1), size(sample, 2), 1]);
else
    % 3D: permute (C, H, W) -> (H, W, C)
    sample = single(permute(sample, [2, 3, 1]));
end
end

function r2 = computeR2(y_true, y_pred)
ss_res = sum((y_true - y_pred).^2, 'all');
ss_tot = sum((y_true - mean(y_true, 'all')).^2, 'all');
r2 = 1 - (ss_res / ss_tot);
end
