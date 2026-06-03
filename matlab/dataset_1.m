% generate_dataset_balanced1920_sequence_features.m
% Corrected IEEE 33-bus fault dataset generator
% Method: sequence-network fault analysis using Y-bus/Z-bus matrices.
%
% Output: fault_dataset_balanced1920_sequence_features.csv
%
% What this script guarantees:
%   1) Bus 1 is the ideal slack/source bus and remains 1.0 pu.
%   2) Faults are injected through sequence-network equations, not by
%      approximating faults as load changes.
%   3) LG, LL, LLG, LLLG, and HIF cases are generated for buses 2..33.
%   4) Output columns include voltage magnitude, voltage angle, and
%      sequence-component magnitudes:
%      V_bus1 ... V_bus33, Angle_bus1 ... Angle_bus33,
%      V0_bus1 ... V0_bus33, V1_bus1 ... V1_bus33, V2_bus1 ... V2_bus33,
%      fault_type, fault_bus, fault_resistance
%
% Notes for EE validation:
%   - LG = single line-to-ground fault on phase A.
%   - LL = line-to-line fault between phases B and C.
%   - LLG = double line-to-ground fault on phases B and C.
%   - LLLG = balanced three-phase-to-ground fault.
%   - HIF = high-impedance LG fault.
%   - Reported bus voltage is the minimum phase-voltage magnitude at that bus.
%     This is intentional because it captures the voltage sag caused by faults.
%
% Run in MATLAB:
%   >> generate_dataset_balanced1920_sequence_features
%
% Author: project script for AI fault detection dataset generation

clear; clc;

%% -------------------- User settings --------------------
outFile = 'fault_dataset_balanced1920_sequence_features.csv';

% IEEE 33-bus distribution feeder base values
baseMVA = 100;       % MVA, standard per-unit base
baseKV  = 12.66;     % kV line-to-line
Zbase   = (baseKV^2) / baseMVA;   % ohm

nBus = 33;
slackBus = 1;
faultBuses = 2:33;

% Fault resistances in pu.
% Fault rows: 32 buses * (5 fault types * 10 Rf) = 1600 rows.
% Normal rows: generated from 320 different healthy loading scenarios.
% Total rows: 1920. This fixes the old one-Normal-row class problem
% and keeps all six classes balanced at 320 rows each.
rfLow = [0 0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0];
rfHIF = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];

% Number of healthy/no-fault examples. Keep this close to the main fault classes
% so the model learns what normal operation looks like.
nNormalScenarios = 320;

% Sequence impedance assumptions.
% For most distribution lines, Z2 is close to Z1.
% Z0 is usually larger because zero-sequence return paths include earth/neutral.
Z2_multiplier = 1.00;
Z0_multiplier = 3.00;

% Numerical safety value to avoid division by exactly zero.
epsZ = 1e-9;

%% -------------------- IEEE 33-bus line data --------------------
% Format: fromBus, toBus, R_ohm, X_ohm
% Standard Baran-Wu IEEE 33-bus radial distribution feeder data.
lineData = [
     1   2   0.0922   0.0470;
     2   3   0.4930   0.2511;
     3   4   0.3660   0.1864;
     4   5   0.3811   0.1941;
     5   6   0.8190   0.7070;
     6   7   0.1872   0.6188;
     7   8   0.7114   0.2351;
     8   9   1.0300   0.7400;
     9  10   1.0440   0.7400;
    10  11   0.1966   0.0650;
    11  12   0.3744   0.1238;
    12  13   1.4680   1.1550;
    13  14   0.5416   0.7129;
    14  15   0.5910   0.5260;
    15  16   0.7463   0.5450;
    16  17   1.2890   1.7210;
    17  18   0.7320   0.5740;
     2  19   0.1640   0.1565;
    19  20   1.5042   1.3554;
    20  21   0.4095   0.4784;
    21  22   0.7089   0.9373;
     3  23   0.4512   0.3083;
    23  24   0.8980   0.7091;
    24  25   0.8960   0.7011;
     6  26   0.2030   0.1034;
    26  27   0.2842   0.1447;
    27  28   1.0590   0.9337;
    28  29   0.8042   0.7006;
    29  30   0.5075   0.2585;
    30  31   0.9744   0.9630;
    31  32   0.3105   0.3619;
    32  33   0.3410   0.5302
];

%% -------------------- Load data for prefault voltages --------------------
% Format: bus, P_kW, Q_kVAr
% Standard IEEE 33-bus loads. Bus 1 has no load.
loadData = [
     1     0     0;
     2   100    60;
     3    90    40;
     4   120    80;
     5    60    30;
     6    60    20;
     7   200   100;
     8   200   100;
     9    60    20;
    10    60    20;
    11    45    30;
    12    60    35;
    13    60    35;
    14   120    80;
    15    60    10;
    16    60    20;
    17    60    20;
    18    90    40;
    19    90    40;
    20    90    40;
    21    90    40;
    22    90    40;
    23    90    50;
    24   420   200;
    25   420   200;
    26    60    25;
    27    60    25;
    28    60    20;
    29   120    70;
    30   200   600;
    31   150    70;
    32   210   100;
    33    60    40
];

%% -------------------- Build sequence Y-bus matrices --------------------
Z1_lines = (lineData(:,3) + 1i*lineData(:,4)) / Zbase;
Z2_lines = Z2_multiplier * Z1_lines;
Z0_lines = Z0_multiplier * Z1_lines;

Y1 = buildYbus(nBus, lineData(:,1), lineData(:,2), Z1_lines);
Y2 = buildYbus(nBus, lineData(:,1), lineData(:,2), Z2_lines);
Y0 = buildYbus(nBus, lineData(:,1), lineData(:,2), Z0_lines);

% Reduced Z-bus matrices with ideal voltage source at Bus 1.
% This is the Y-bus matrix modification/Kron-reduction step.
[Z1red, nonSlack] = reducedZbus(Y1, slackBus);
[Z2red, ~]        = reducedZbus(Y2, slackBus);
[Z0red, ~]        = reducedZbus(Y0, slackBus);

%% -------------------- Prefault voltage profile --------------------
% Balanced prefault load-flow using backward/forward sweep.
% This is only for prefault operating voltage. Faults are NOT modeled as loads.
baseSload = makeBaseSload(nBus, loadData, baseMVA);

% Base-case prefault voltage used for fault studies.
Vpref = radialLoadFlowBFS(nBus, lineData, Z1_lines, baseSload, slackBus);
Vpref(slackBus) = 1 + 0i;

%% -------------------- Dataset generation --------------------
rows = {};

% Healthy/no-fault rows.
% These are NOT copied duplicates. Each row is a separate normal operating
% point created by changing the load level and bus-to-bus loading pattern,
% then solving the feeder load-flow again.
for scenario = 1:nNormalScenarios
    normalSload = makeNormalScenarioLoad(baseSload, scenario, nNormalScenarios, slackBus);
    Vnormal = radialLoadFlowBFS(nBus, lineData, Z1_lines, normalSload, slackBus);
    Vnormal(slackBus) = 1 + 0i;

    % Healthy balanced operation: positive sequence only.
    normalV0 = zeros(nBus,1);
    normalV1 = Vnormal(:);
    normalV2 = zeros(nBus,1);

    normalVoltages = abs(normalV1).';
    normalAngles = rad2deg(angle(normalV1)).';
    normalSeq0 = abs(normalV0).';
    normalSeq1 = abs(normalV1).';
    normalSeq2 = abs(normalV2).';

    normalVoltages(1) = 1.0;
    normalAngles(1) = 0.0;
    normalSeq0(1) = 0.0;
    normalSeq1(1) = 1.0;
    normalSeq2(1) = 0.0;

    % fault_bus = 0 and fault_resistance = 0 mean no fault.
    rows(end+1,:) = makeRow(normalVoltages, normalAngles, normalSeq0, normalSeq1, normalSeq2, 'Normal', 0, 0); %#ok<SAGROW>
end

faultTypes = {'LG','LL','LLG','LLLG','HIF'};

for fb = faultBuses
    for ftIdx = 1:numel(faultTypes)
        ft = faultTypes{ftIdx};
        if strcmp(ft, 'HIF')
            rfList = rfHIF;
        else
            rfList = rfLow;
        end

        for rf = rfList
            [Vabc, V0, V1, V2] = solveFaultSequence(fb, ft, rf, Vpref, Z1red, Z2red, Z0red, nonSlack, slackBus, nBus, epsZ);

            % Use minimum phase magnitude per bus as ML voltage-sag feature.
            Vmag = min(abs(Vabc), [], 2).';

            % Export positive-sequence voltage angle and sequence magnitudes.
            Vangle = rad2deg(angle(V1)).';
            V0mag = abs(V0).';
            V1mag = abs(V1).';
            V2mag = abs(V2).';

            % Enforce ideal slack/source values. This is physically intentional.
            Vmag(1) = 1.0;
            Vangle(1) = 0.0;
            V0mag(1) = 0.0;
            V1mag(1) = 1.0;
            V2mag(1) = 0.0;

            rows(end+1,:) = makeRow(Vmag, Vangle, V0mag, V1mag, V2mag, ft, fb, rf); %#ok<SAGROW>
        end
    end
end

varNames = cell(1, 5*nBus + 3);
col = 1;
for b = 1:nBus
    varNames{col} = sprintf('V_bus%d', b);
    col = col + 1;
end
for b = 1:nBus
    varNames{col} = sprintf('Angle_bus%d', b);
    col = col + 1;
end
for b = 1:nBus
    varNames{col} = sprintf('V0_bus%d', b);
    col = col + 1;
end
for b = 1:nBus
    varNames{col} = sprintf('V1_bus%d', b);
    col = col + 1;
end
for b = 1:nBus
    varNames{col} = sprintf('V2_bus%d', b);
    col = col + 1;
end
varNames{col} = 'fault_type';
varNames{col+1} = 'fault_bus';
varNames{col+2} = 'fault_resistance';

T = cell2table(rows, 'VariableNames', varNames);
writetable(T, outFile);

%% -------------------- Built-in validation report --------------------
fprintf('\nSaved: %s\n', outFile);
fprintf('Rows generated: %d\n', height(T));
fprintf('Columns generated: %d = 33 Vmag + 33 angle + 33 V0 + 33 V1 + 33 V2 + 3 labels\n', width(T));
fprintf('Expected rows: 1920 = 320 rows for each of 6 classes\n');

% 1. Clean reference to the table variable directly
bus1Vals = T.V_bus1;
fprintf('Bus 1 min/max voltage: %.6f / %.6f pu\n', min(bus1Vals), max(bus1Vals));

expectedTypes = {'Normal','LG','LL','LLG','LLLG','HIF'};
fprintf('Class counts:\n');
for ci = 1:numel(expectedTypes)
    c = sum(strcmp(T.fault_type, expectedTypes{ci}));
    fprintf('  %s: %d\n', expectedTypes{ci}, c);
end

% 2. FIXED LINE HERE: Removed the cell2mat wrapper around fault_resistance
checkRows = T(strcmp(T.fault_type,'LLLG') & T.fault_resistance==0, :);
if ~isempty(checkRows)
    vFault = zeros(height(checkRows),1);
    for i = 1:height(checkRows)
        fb = checkRows.fault_bus(i); % Changed curly brackets to smooth indexing
        vFault(i) = checkRows{i, sprintf('V_bus%d', fb)}; % Direct array value grab
    end
    fprintf('LLLG bolted fault-bus voltage range: %.6f to %.6f pu\n', min(vFault), max(vFault));
end

fprintf('\nValidation checklist for EE student:\n');
fprintf('  [1] Rows should equal 1920.\n');
fprintf('  [1b] Every class count should equal 320, including Normal and HIF.\n');
fprintf('  [2] V_bus1 should be exactly 1.0 pu for every row.\n');
fprintf('  [2b] Angle_bus1=0, V0_bus1=0, V1_bus1=1, V2_bus1=0 for every row.\n');
fprintf('  [3] Bolted LLLG fault-bus voltages should be near zero.\n');
fprintf('  [4] Higher fault resistance should generally cause less severe voltage drop.\n');
fprintf('  [5] HIF rows should show smaller voltage drops than low-resistance LG faults.\n\n');

%% ==================== Local functions ====================
function Sload = makeBaseSload(nBus, loadData, baseMVA)
    Sload = zeros(nBus,1);
    for r = 1:size(loadData,1)
        b = loadData(r,1);
        Sload(b) = (loadData(r,2) + 1i*loadData(r,3)) / (baseMVA*1000); % pu
    end
end

function Sscenario = makeNormalScenarioLoad(baseSload, scenario, nScenarios, slackBus)
    % Creates deterministic healthy operating points.
    % No random numbers are used, so every EE student gets the same CSV.
    nBus = length(baseSload);

    % Overall feeder loading varies from light to heavy load.
    globalScale = 0.55 + 0.90 * (scenario - 1) / max(nScenarios - 1, 1);  % 0.55..1.45

    % Small bus-level diversity avoids 320 nearly identical normal rows.
    busIndex = (1:nBus).';
    shape1 = 0.08 * sin(2*pi*scenario/37 + busIndex/5);
    shape2 = 0.05 * cos(2*pi*scenario/53 + busIndex/7);
    busScale = 1 + shape1 + shape2;

    % Keep every individual load in a realistic range.
    busScale = max(0.75, min(1.25, busScale));

    Sscenario = baseSload .* globalScale .* busScale;
    Sscenario(slackBus) = 0;
end

function Y = buildYbus(nBus, fromBus, toBus, Zline)
    Y = complex(zeros(nBus, nBus));
    for i = 1:length(Zline)
        f = fromBus(i);
        t = toBus(i);
        y = 1 / Zline(i);
        Y(f,f) = Y(f,f) + y;
        Y(t,t) = Y(t,t) + y;
        Y(f,t) = Y(f,t) - y;
        Y(t,f) = Y(t,f) - y;
    end
end

function [Zred, nonSlack] = reducedZbus(Y, slackBus)
    nBus = size(Y,1);
    nonSlack = setdiff(1:nBus, slackBus);
    Yred = Y(nonSlack, nonSlack);
    Zred = inv(Yred);
end

function V = radialLoadFlowBFS(nBus, lineData, Zline, Sload, slackBus)
    maxIter = 200;
    tol = 1e-10;
    V = ones(nBus,1);

    from = lineData(:,1);
    to = lineData(:,2);
    nLine = length(from);

    parent = zeros(nBus,1);
    branchOfBus = zeros(nBus,1);
    children = cell(nBus,1);
    for ell = 1:nLine
        parent(to(ell)) = from(ell);
        branchOfBus(to(ell)) = ell;
        children{from(ell)} = [children{from(ell)}, to(ell)]; %#ok<AGROW>
    end

    order = bfsOrder(children, slackBus);
    revOrder = fliplr(order);

    for iter = 1:maxIter
        Vold = V;
        Iinj = zeros(nBus,1);
        for b = 1:nBus
            if b ~= slackBus
                if abs(V(b)) < 1e-6
                    V(b) = 1;
                end
                Iinj(b) = conj(Sload(b) / V(b));
            end
        end

        Ibranch = zeros(nLine,1);
        Isum = Iinj;
        for b = revOrder
            if b == slackBus
                continue;
            end
            ell = branchOfBus(b);
            Ibranch(ell) = Isum(b);
            p = parent(b);
            Isum(p) = Isum(p) + Ibranch(ell);
        end

        V(slackBus) = 1 + 0i;
        for b = order
            if b == slackBus
                continue;
            end
            p = parent(b);
            ell = branchOfBus(b);
            V(b) = V(p) - Zline(ell) * Ibranch(ell);
        end

        if max(abs(V - Vold)) < tol
            break;
        end
    end
end

function order = bfsOrder(children, root)
    order = root;
    q = root;
    head = 1;
    while head <= numel(q)
        node = q(head);
        head = head + 1;
        c = children{node};
        for k = 1:numel(c)
            q(end+1) = c(k); %#ok<AGROW>
            order(end+1) = c(k); %#ok<AGROW>
        end
    end
end

function [Vabc, V0, V1, V2] = solveFaultSequence(faultBus, faultType, Rf, Vpref, Z1red, Z2red, Z0red, nonSlack, slackBus, nBus, epsZ)
    % Returns nBus x 3 complex phase voltages [Va Vb Vc].
    % Sequence order: V0, V1, V2 are converted to phase values.

    a = exp(1i*2*pi/3);
    A = [1 1 1; 1 a^2 a; 1 a a^2];

    V0 = zeros(nBus,1);
    V1 = Vpref(:);
    V2 = zeros(nBus,1);

    if faultBus == slackBus
    Vabc = [ones(nBus,1), ones(nBus,1)*a^2, ones(nBus,1)*a];
    V0 = zeros(nBus,1);
    V1 = Vpref(:);
    V2 = zeros(nBus,1);
    return;
end

    k = find(nonSlack == faultBus, 1);
    if isempty(k)
        error('Fault bus was not found in non-slack bus list.');
    end

    Z1kk = Z1red(k,k) + epsZ;
    Z2kk = Z2red(k,k) + epsZ;
    Z0kk = Z0red(k,k) + epsZ;
    Vf = Vpref(faultBus);
    Zf = Rf + 0i;

    switch upper(faultType)
        case {'LG','HIF'}
            % Single line-to-ground fault: sequence networks in series.
            I1 = Vf / (Z1kk + Z2kk + Z0kk + 3*Zf);
            I2 = I1;
            I0 = I1;

        case 'LL'
            % Line-to-line fault: positive and negative sequence networks only.
            I1 = Vf / (Z1kk + Z2kk + Zf);
            I2 = -I1;
            I0 = 0;

        case 'LLG'
            % Double line-to-ground fault: Z2 in parallel with Z0 + 3Zf.
            Z0f = Z0kk + 3*Zf;
            Zpar = (Z2kk * Z0f) / (Z2kk + Z0f);
            I1 = Vf / (Z1kk + Zpar);
            I2 = -I1 * (Z0f / (Z2kk + Z0f));
            I0 = -I1 * (Z2kk / (Z2kk + Z0f));

        case 'LLLG'
            % Balanced three-phase-to-ground fault: only positive sequence.
            I1 = Vf / (Z1kk + Zf);
            I2 = 0;
            I0 = 0;

        otherwise
            error('Unknown fault type: %s', faultType);
    end

    idx = nonSlack;
    V1(idx) = Vpref(idx) - Z1red(:,k) * I1;
    V2(idx) =          - Z2red(:,k) * I2;
    V0(idx) =          - Z0red(:,k) * I0;

    % Ideal slack/source bus remains balanced at 1.0 pu.
    V1(slackBus) = 1 + 0i;
    V2(slackBus) = 0 + 0i;
    V0(slackBus) = 0 + 0i;

    Vabc = zeros(nBus,3);
    for b = 1:nBus
        seq = [V0(b); V1(b); V2(b)];
        ph = A * seq;
        Vabc(b,:) = ph.';
    end
end

function row = makeRow(Vmag, Vangle, V0mag, V1mag, V2mag, faultType, faultBus, Rf)
    nBus = numel(Vmag);
    row = cell(1, 5*nBus + 3);
    col = 1;
    for b = 1:nBus
        row{col} = double(real(Vmag(b)));
        col = col + 1;
    end
    for b = 1:nBus
        row{col} = double(real(Vangle(b)));
        col = col + 1;
    end
    for b = 1:nBus
        row{col} = double(real(V0mag(b)));
        col = col + 1;
    end
    for b = 1:nBus
        row{col} = double(real(V1mag(b)));
        col = col + 1;
    end
    for b = 1:nBus
        row{col} = double(real(V2mag(b)));
        col = col + 1;
    end
    row{col} = faultType;
    row{col+1} = faultBus;
    row{col+2} = Rf;
end
