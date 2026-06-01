clear; clc;

nBus = 33;
Vbase = 12660;
Sbase = 1e6;
Zbase = Vbase^2 / Sbase;

Pload = [0,100,90,120,60,60,200,200,60,60,45,60,60,120,60,60,60,90,90,90,90,90,90,420,420,60,60,60,120,200,150,210,60]/1000;
Qload = [0,60,40,80,30,20,100,100,20,20,30,35,35,80,10,20,20,40,40,40,40,40,50,200,200,25,25,20,70,600,70,100,40]/1000;

branch = [
    1,2,0.0922,0.0470; 2,3,0.4930,0.2511; 3,4,0.3660,0.1864;
    4,5,0.3811,0.1941; 5,6,0.8190,0.7070; 6,7,0.1872,0.6188;
    7,8,0.7114,0.2351; 8,9,1.0300,0.7400; 9,10,1.0440,0.7400;
    10,11,0.1966,0.0650; 11,12,0.3744,0.1238; 12,13,1.4680,1.1550;
    13,14,0.5416,0.7129; 14,15,0.5910,0.5260; 15,16,0.7463,0.5450;
    16,17,1.2890,1.7210; 17,18,0.7320,0.5740; 2,19,0.1640,0.1565;
    19,20,1.5042,1.3554; 20,21,0.4095,0.4784; 21,22,0.7089,0.9373;
    3,23,0.4512,0.3083; 23,24,0.8980,0.7091; 24,25,0.8960,0.7011;
    6,26,0.2030,0.1034; 26,27,0.2842,0.1447; 27,28,1.0590,0.9337;
    28,29,0.8042,0.7006; 29,30,0.5075,0.2585; 30,31,0.9744,0.9630;
    31,32,0.3105,0.3619; 32,33,0.3410,0.5302;
];

branch(:,3) = branch(:,3)/Zbase;
branch(:,4) = branch(:,4)/Zbase;

% Fault settings
faultTypes = {'Normal','LG','LL','LLG','LLLG','HIF'};
faultResistances = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];

% Build header
header = cell(1, nBus + 3);
for i = 1:nBus
    header{i} = sprintf('V_bus%d', i);
end
header{nBus+1} = 'fault_type';
header{nBus+2} = 'fault_bus';
header{nBus+3} = 'fault_resistance';

allData = {};
rowCount = 0;

fprintf('Generating dataset...\n');

for fi = 1:length(faultTypes)
    fType = faultTypes{fi};
    
    if strcmp(fType, 'Normal')
        busLoop = 1;
        rfLoop = 0;
    elseif strcmp(fType, 'HIF')
        busLoop = 2:33;
        rfLoop = faultResistances(6:10); % only high resistance for HIF
    else
        busLoop = 2:33;
        rfLoop = faultResistances;
    end
    
    for fBus = busLoop
        for rf = rfLoop
            % Run load flow with fault
            Pf = Pload;
            Qf = Qload;
            
            % Fault effect: drop voltage at fault bus by adding load
            if strcmp(fType, 'Normal')
                % no change
            elseif strcmp(fType, 'LG')
                Pf(fBus) = Pf(fBus) + (1^2)/(rf + 0.001) * 0.33;
                Qf(fBus) = Qf(fBus) + (1^2)/(rf + 0.001) * 0.1;
            elseif strcmp(fType, 'LL')
                Pf(fBus) = Pf(fBus) + (1^2)/(rf + 0.001) * 0.58;
                Qf(fBus) = Qf(fBus) + (1^2)/(rf + 0.001) * 0.2;
            elseif strcmp(fType, 'LLG')
                Pf(fBus) = Pf(fBus) + (1^2)/(rf + 0.001) * 0.75;
                Qf(fBus) = Qf(fBus) + (1^2)/(rf + 0.001) * 0.3;
            elseif strcmp(fType, 'LLLG')
                Pf(fBus) = Pf(fBus) + (1^2)/(rf + 0.001) * 1.0;
                Qf(fBus) = Qf(fBus) + (1^2)/(rf + 0.001) * 0.5;
            elseif strcmp(fType, 'HIF')
                Pf(fBus) = Pf(fBus) + (1^2)/(rf + 0.001) * 0.1;
                Qf(fBus) = Qf(fBus) + (1^2)/(rf + 0.001) * 0.05;
            end
            
            % Run backward-forward sweep
            V = ones(nBus,1);
            nBranch = size(branch,1);
            
            for iter = 1:100
                V_old = V;
                Ibr = zeros(nBranch,1);
                
                for k = nBranch:-1:1
                    t = branch(k,2);
                    S = Pf(t) + 1j*Qf(t);
                    Ibr(k) = conj(S / V(t));
                    for m = 1:nBranch
                        if branch(m,1) == t
                            Ibr(k) = Ibr(k) + Ibr(m);
                        end
                    end
                end
                
                V(1) = 1.0;
                for k = 1:nBranch
                    f = branch(k,1);
                    t = branch(k,2);
                    Z = branch(k,3) + 1j*branch(k,4);
                    V(t) = V(f) - Z*Ibr(k);
                end
                
                if max(abs(abs(V)-abs(V_old))) < 1e-6
                    break
                end
            end
            
            Vmag = abs(V)';
            row = [num2cell(Vmag), {fType}, {fBus}, {rf}];
            allData = [allData; row];
            rowCount = rowCount + 1;
        end
    end
    fprintf('Done: %s (%d rows so far)\n', fType, rowCount);
end

% Write to CSV
T = cell2table(allData, 'VariableNames', header);
writetable(T, 'fault_dataset.csv');
fprintf('\nDataset saved: fault_dataset.csv\n');
fprintf('Total rows: %d\n', rowCount);