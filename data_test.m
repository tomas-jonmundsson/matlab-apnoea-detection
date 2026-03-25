%% === Build submission for independent test set (EPOCH variant: 18 features) ===
% Mirrors LORO_epoch training: tri-epoch features [-30:0, 0:30, 30:60]
% Labels are produced at 1 Hz; per-second decision anchored at end of 30:60 window.

%% 0) Train on COMPLETE TRAINING SET (if not already in workspace)
% Expect X and T from the LORO_epoch training build; otherwise rebuild from it.
if ~exist('X_all','var') || ~exist('T_all','var')
    if ~exist('X','var') || ~exist('T','var')
        error('X/T not in workspace. Run your LORO_epoch feature build first.');
    end
    X_all = cat(1, X{:});
    T_all = cat(1, T{:});
end

% Linear regression head on ALL data (kept exactly as in training file)
W_full = TrainLR(X_all, T_all);

% Full-set threshold using PR max-F1 on ALL scores (unchanged)
[~, Y_all] = PredLin(W_full, X_all);
a_all = Y_all(:,1);
y_all = T_all(:,1) > 0;
thr_full = choose_threshold_on_train(a_all, y_all);
fprintf('\n[Submission] Full-set model trained. Chosen threshold = %.4f\n', thr_full);

%% 1) Load TEST data + template annotations (for shape and 1 Hz length)
load('ProjectTestData.mat');        % ECG, QRS, SpO2 for 100 test records
load('ProjectTestAnnotations.mat'); % Class with '?' placeholders (1 Hz)
Fs = 200;
WindowLen = Fs * 30;    % 30 s
Step      = Fs *  1;    % 1 s hop

%% 2) Clean SpO2 on TEST exactly like TRAIN ([50,100] -> NaN)
SpO2_clean_test = cell(size(SpO2));
for i = 1:numel(SpO2)
    v = double(SpO2{i}(:));
    v(v < 50 | v > 100) = NaN;
    SpO2_clean_test{i} = v;
end

%% 3) Predict per-second labels using tri-epoch 18-D features
Class_out = Class;   % copy template to preserve record count and seconds

for p = 1:numel(ECG)
    ecg = ECG{p};
    qrs = QRS{p}(:);
    sp  = SpO2_clean_test{p};
    N   = numel(ecg);

    % Per-second template length
    Lsec = numel(Class{p});
    if N < 3*WindowLen || Lsec == 0
        % too short or empty -> default all 'N' (as column vector)
        Class_out{p} = repmat('N', Lsec, 1);
        continue
    end

    % RR (seconds) and second-beat timestamps (samples); keep alignment, mark OOR as NaN
    rr   = diff(qrs)/Fs;
    rr_t = qrs(2:end);
    bad  = (rr < 0.3) | (rr > 3.0);
    rr(bad) = NaN;

    % Anchor at start of 0:30; need future window -> 3*WindowLen coverage
    Nwin = floor((N - 3*WindowLen)/Step) + 1;

    % Accumulate A-scores by **second** where the FUTURE window ends
    score_sum = zeros(Lsec,1);
    score_cnt = zeros(Lsec,1);

    for j = 1:Nwin
        seg0_start = (j-1)*Step + 1;             % 0:30 start
        seg0_end   = seg0_start + WindowLen - 1; % 0:30 end

        segP_start = seg0_start - WindowLen;     % -30:0
        segP_end   = seg0_start - 1;

        segF_start = seg0_end + 1;               % 30:60
        segF_end   = seg0_end + WindowLen;

        % Extract 6 features for each epoch; concat -> 18-D
        fPast  = local_feats(rr, rr_t, sp, Fs, segP_start, segP_end);
        fCurr  = local_feats(rr, rr_t, sp, Fs, seg0_start, seg0_end);
        fFutr  = local_feats(rr, rr_t, sp, Fs, segF_start, segF_end);
        xj = [fPast, fCurr, fFutr];              % 1×18

        if any(isnan(xj))
            continue; % invalid window; skip
        end

        % Score with full model; take A-score
        [~, Yj] = PredLin(W_full, xj);
        aScore = Yj(1);

        % Second index is END of FUTURE window (consistent with training labels)
        secIdx = floor(segF_end / Fs);
        if secIdx >= 1 && secIdx <= Lsec && isfinite(aScore)
            score_sum(secIdx) = score_sum(secIdx) + aScore;
            score_cnt(secIdx) = score_cnt(secIdx) + 1;
        end
    end

    % Per-second decision: mean A-score across windows that land in that second
    mean_score = zeros(Lsec,1);
    nz = score_cnt > 0;
    mean_score(nz) = score_sum(nz) ./ score_cnt(nz);
    mean_score(~nz) = -inf;  % no supporting window -> default below threshold

    yhat_sec = mean_score >= thr_full;

    % Output Lsec-by-1 char column
    lab = repmat('N', Lsec, 1);
    lab(yhat_sec) = 'A';
    Class_out{p} = lab;
end

% Enforce column orientation for every record (belt-and-braces)
for p = 1:numel(Class_out)
    if isrow(Class_out{p})
        Class_out{p} = Class_out{p}.';
    end
end

%% 4) Save ONLY annotations, in the required format
submission_name = 'ProjectTestAnnotations_Group17_Submission2.mat';  % TODO: set your group
Class = Class_out; %#ok<NASGU>
save(submission_name, 'Class');
fprintf('[Submission] Wrote %s (Class at 1 Hz, %d records)\n', submission_name, numel(Class));

%% === Local helpers (PredLin PRESERVED EXACTLY) ============================
function f6 = local_feats(rr, rr_t, spo2_1hz, Fs, sampStart, sampEnd)
    % RR slice by second-beat timestamps in SAMPLE domain (keep NaN filter local)
    mask  = (rr_t >= sampStart) & (rr_t <= sampEnd);
    RRseg = rr(mask);
    RRseg = RRseg(isfinite(RRseg));  % drop NaNs only here

    % SpO2 slice in 1 Hz
    tStart = floor(sampStart/Fs) + 1;
    tEnd   = floor(sampEnd  /Fs);
    okS = (tStart >= 1) && (tEnd <= numel(spo2_1hz)) && (tStart <= tEnd);

    if numel(RRseg) >= 2 && okS
        dRR    = diff(RRseg);
        NN50   = sum(abs(dRR) > 0.050);
        SDSD   = std(dRR,0,'omitnan');
        meanRR = mean(RRseg,'omitnan');
        stdRR  = std( RRseg,0,'omitnan');
        sseg   = spo2_1hz(tStart:tEnd);
        mS     = mean(sseg,'omitnan');
        sdS    = std( sseg,0,'omitnan');
        f6 = [meanRR, stdRR, NN50, SDSD, mS, sdS];
    else
        f6 = nan(1,6);
    end
end

function [T,Y]=PredLin(W,X)
%Predict with linear classifier (kept exactly)
K=size(X,1); N=size(W,2);
X=[X,ones(K,1)];      % add bias
Y=X*W;                % scores
[~,temp]=max(Y,[],2);
T=zeros(K,N);
for i=1:K, T(i,temp(i))=1; end
end

function W=TrainLR(X,T)
%Train a linear regression classifier using data (X) and targets (T)
K=size(X,1);
X=[X,ones(K,1)];      % add bias
W=(X'*X)\(X'*T);      % closed-form LS
end

function thr = choose_threshold_on_train(aScore, yTrue)
% PR max-F1 threshold; robust to single-class edge cases
u = unique(yTrue);
if numel(u) < 2
    thr = (all(yTrue==0)) * (max(aScore)+eps) + (all(yTrue==1)) * (min(aScore)-eps);
    return
end
cuts = unique(aScore);
if numel(cuts) > 512
    cuts = unique(quantile(aScore, linspace(0,1,512)));
end
bestF1 = -inf; thr = median(aScore,'omitnan');
for c = cuts(:)'
    yhat = aScore >= c;
    TP = sum( yhat &  yTrue);
    FP = sum( yhat & ~yTrue);
    FN = sum(~yhat &  yTrue);
    denom = (2*TP + FP + FN);
    F1c = (denom==0) * 0 + (denom>0) * (2*TP/denom);
    if F1c > bestF1, bestF1 = F1c; thr = c; end
end
end
