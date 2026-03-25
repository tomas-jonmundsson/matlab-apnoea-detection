%% === LOAD FIRST ===
load("ProjectTrainData.mat");   % ECG, QRS, Class, SpO2 (cell arrays)

%% === Parameters ===
Fs        = 200;        % Hz
Npatients = 100;        % number of patients to use
epochSec  = 60;         % labels per minute
winSec    = 30;         % feature window: t : t+30 s
EpochLen  = Fs * epochSec;
WinLen    = Fs * winSec;

F = min(Npatients, numel(ECG));

%% === Feature config (ADD THIS ONCE NEAR PARAMETERS) ===
FEATURES = { ...
    'meanRR','stdRR','NN50','SDSD', ...        % original
    'meanSpO2','stdSpO2', ...                  % original
    'pNN50','RMSSD','medianRR','iqrRR','logStdRR','logSDSD','meanHR', ...
    % 'minSpO2','pctLT90','spo2Slope', ...
    % 'LF','HF','LFHF','TotPow' ...
};
% % Freq-HRV settings (only used if LF/HF/LFHF/TotPow present)
% fs_rr   = 4;                      % Hz
% lf_band = [0.04 0.15];            % Hz
% hf_band = [0.15 0.40];            % Hz


%% 0) Clean SpO2
SpO2_clean = cell(size(SpO2));
for i = 1:Npatients
    v = double(SpO2{i}(:));
    v(v < 50 | v > 100) = NaN;
    SpO2_clean{i} = v;
end

%% === RR build (seconds) + RR midpoints (samples) ===
RR = cell(F,1); RRmid = cell(F,1);
for p = 1:F
    q = QRS{p}(:);
    if numel(q) < 2
        RR{p} = []; RRmid{p} = [];
        continue
    end
    rr  = diff(q)/Fs;                    % seconds
    bad = (rr < 0.3) | (rr > 3.0);       % out-of-range
    rr(bad) = NaN;                       % <-- mark as NaN, keep alignment

    qm = (q(1:end-1) + q(2:end))/2;      % samples (midpoint timestamp)

    RR{p}   = rr;                        % <-- keep NaNs (no filtering here)
    RRmid{p}= qm;                        % <-- keep all midpoints (no subsetting)
end

%% 1) Feature extraction (3 epochs × selected features)
WindowLen = Fs * 30;     % 30 s
Step      = Fs *  1;     % 1 s hop

X = cell(Npatients,1);
T = cell(Npatients,1);

for p = 1:Npatients
    ecg  = ECG{p};
    qrs  = QRS{p}(:);
    Nsamples = numel(ecg);

    if Nsamples < 3*WindowLen           % need room for -30:0, 0:30, 30:60
        X{p} = []; T{p} = []; continue;
    end

    % RR & timestamp (second beat), with out-of-range -> NaN (keep alignment)
    rr   = diff(qrs)/Fs; 
    rr_t = qrs(2:end);
    bad  = (rr < 0.3) | (rr > 3.0);
    rr(bad) = NaN;

    Nwin = floor((Nsamples - 3*WindowLen)/Step) + 1;   % anchor at start of 0:30
    FeatPerEpoch = numel(FEATURES);
    Xi = nan(Nwin, 3*FeatPerEpoch);                    % 3 epochs × FeatPerEpoch
    Ti = zeros(Nwin, 2);                               % [A N]

    % helper to compute features for an RR/SpO2 window in samples
    feats = @(sampA,sampB) local_feats_flex( ...
        rr, rr_t, SpO2_clean{p}, Fs, sampA, sampB, ...
        FEATURES);

    for j = 1:Nwin
        seg0_start = (j-1)*Step + 1;               % 0:30 start
        seg0_end   = seg0_start + WindowLen - 1;   % 0:30 end

        segP_start = seg0_start - WindowLen;       % -30:0
        segP_end   = seg0_start - 1;

        segF_start = seg0_end + 1;                 % 30:60
        segF_end   = seg0_end + WindowLen;

        fPast   = feats(segP_start, segP_end);
        fCurr   = feats(seg0_start, seg0_end);
        fFutur  = feats(segF_start, segF_end);

        Xi(j,:) = [fPast, fCurr, fFutur];

        % Label from END of 30:60 epoch (future framing)
        lblIdx = floor(segF_end / Fs);
        if lblIdx <= numel(Class{p})
            c = Class{p}(lblIdx);
            if     c=='A', Ti(j,:) = [1 0];
            elseif c=='N', Ti(j,:) = [0 1];
            end
        end
    end

    % Drop rows with any NaN OR unlabeled (exactly like before)
    keep = ~any(isnan(Xi),2) & any(Ti,2);
    X{p} = Xi(keep,:);
    T{p} = Ti(keep,:);
end


% % --- PR collectors (outer-test) ---
% allScores = [];    % will size after cvpartition
% allTruth  = [];
% After building X{p} inside your feature extraction:
% store total candidate windows per patient
Nwin_raw = zeros(Npatients,1);
for p = 1:Npatients
    ecg = ECG{p};
    Nsamples = numel(ecg);

    if Nsamples < 3*WindowLen
        Nwin_raw(p) = 0;
        continue;
    end

    Nwin_raw(p) = floor((Nsamples - 3*WindowLen)/Step) + 1;  % same as in loop
end

% Now compute survival rates using the final X{p}
survival_pct = nan(Npatients,1);
for p = 1:Npatients
    if Nwin_raw(p) > 0
        survival_pct(p) = 100 * (size(X{p},1) / Nwin_raw(p));
    end
end

figure('Color','w','Position',[120 120 800 360]);
histogram(survival_pct, 10);
xlabel('Surviving epochs (%)');
ylabel('Number of patients');
title('Distribution of epoch survival across patients');
xlim([0 100]);
grid on;

%% === 10-fold CV (patient-level): train-fit, train-threshold, test-eval ===
cv = cvpartition(F, 'KFold', 10);   % patient-wise partition

TP=0; FP=0; FN=0;
TP_tr=0; FP_tr=0; FN_tr=0;

allScores = cell(cv.NumTestSets,1);
allTruth  = cell(cv.NumTestSets,1);

for fold = 1:cv.NumTestSets
    trMask = training(cv, fold);           % logical mask over patients
    teMask = test(cv, fold);

    trIdx  = find(trMask);
    teIdx  = find(teMask);

    % Aggregate TRAIN/TEST rows across the patients in each fold
    Xtr = cat(1, X{trIdx});   Ttr = cat(1, T{trIdx});
    Xte = cat(1, X{teIdx});   Tte = cat(1, T{teIdx});

    if isempty(Xtr) || isempty(Ttr) || isempty(Xte) || isempty(Tte)
        allScores{fold} = []; allTruth{fold} = [];
        continue
    end

    % Fit linear regression on RAW TRAIN
    W = TrainLR(Xtr, Ttr);                 % scores in [A N]

    % Choose threshold on TRAIN (handles single-class)
    [~, Ytr] = PredLin(W, Xtr);
    aTr = Ytr(:,1);
    yTr = Ttr(:,1) > 0;
    thrA = choose_threshold_on_train(aTr, yTr);

    % --- TRAIN metrics (for accounting only) ---
    yhat_tr = aTr >= thrA;
    TP_tr = TP_tr + sum( yhat_tr &  yTr);
    FP_tr = FP_tr + sum( yhat_tr & ~yTr);
    FN_tr = FN_tr + sum(~yhat_tr &  yTr);

    % --- TEST scoring using TRAIN threshold ---
    [~, Yte] = PredLin(W, Xte);
    aTe  = Yte(:,1);
    yTe  = Tte(:,1) > 0;
    yHat = aTe >= thrA;

    % Collect for outer-test PR
    allScores{fold} = aTe(:);
    allTruth{fold}  = logical(yTe(:));

    % Aggregate fold confusion on TEST
    TP = TP + sum( yHat &  yTe);
    FP = FP + sum( yHat & ~yTe);
    FN = FN + sum(~yHat &  yTe);
end

%% === Train on complete set (100 records) -> Fullset training set results ===
X_all = cat(1, X{:});
T_all = cat(1, T{:});

if ~isempty(X_all) && ~isempty(T_all)
    % Fit linear regression on ALL training data (no normalisation)
    W_full = TrainLR(X_all, T_all);

    % Scores on ALL training data
    [~, Y_all] = PredLin(W_full, X_all);
    a_all = Y_all(:,1);
    y_all = T_all(:,1) > 0;

    % Choose threshold on ALL using the same helper (PR max-F1 sweep)
    thr_full = choose_threshold_on_train(a_all, y_all);

    % In-sample predictions and metrics
    yhat_all = a_all >= thr_full;
    TP_f = sum( yhat_all &  y_all);
    FP_f = sum( yhat_all & ~y_all);
    FN_f = sum(~yhat_all &  y_all);

    Sens_full = 100 * TP_f / max(1,TP_f+FN_f);
    PPV_full  = 100 * TP_f / max(1,TP_f+FP_f);
    F1_full   = 100 * (2*TP_f) / max(1,(2*TP_f + FP_f + FN_f));

    fprintf('\n=== Fullset training set results (trained on 100 records; evaluated in-sample) ===\n');
    fprintf('F1  : %.2f%% | Sens: %.2f%% | PPV: %.2f%% | thr=%.4f\n', F1_full, Sens_full, PPV_full, thr_full);
else
    warning('Full-set training skipped: X_all or T_all is empty.');
end

    % === PPV–Recall curve with F1 maximisation ===
    scores = a_all(:);
    truth  = y_all(:) > 0;        % 1 = A, 0 = N

    % Remove NaNs just in case
    valid = ~isnan(scores);
    scores = scores(valid);
    truth  = truth(valid);

    if ~isempty(scores) && any(truth) && any(~truth)
        % Unique candidate thresholds (descending, high->low)
        thr_grid = unique(scores);
        thr_grid = sort(thr_grid, 'descend');

        nThr  = numel(thr_grid);
        PPV   = zeros(nThr,1);
        Sens  = zeros(nThr,1);
        F1vec = zeros(nThr,1);

        for k = 1:nThr
            thr_k = thr_grid(k);
            yhat  = scores >= thr_k;

            TPk = sum( yhat &  truth);
            FPk = sum( yhat & ~truth);
            FNk = sum(~yhat &  truth);

            if (TPk + FPk) == 0
                PPV(k) = 1;            % define PPV=1 at zero positives
            else
                PPV(k) = TPk / (TPk + FPk);
            end

            if (TPk + FNk) == 0
                Sens(k) = 0;
            else
                Sens(k) = TPk / (TPk + FNk);
            end

            if (PPV(k) + Sens(k)) == 0
                F1vec(k) = 0;
            else
                F1vec(k) = 2 * PPV(k) * Sens(k) / (PPV(k) + Sens(k));
            end
        end

        % Find max-F1 threshold on this curve
        [F1_best, idxBest] = max(F1vec);
        thr_best = thr_grid(idxBest);

        % Plot PPV–Recall curve with max-F1 point highlighted
        figure('Color','w','Position',[140 140 900 420]); hold on; grid on;
        plot(Sens*100, PPV*100, '-', 'LineWidth', 2);           % curve
        plot(Sens(idxBest)*100, PPV(idxBest)*100, 'o', ...
             'MarkerSize', 8, 'LineWidth', 2);                  % best F1

        xlabel('Recall (Sensitivity) [%]');
        ylabel('PPV [%]');
        title(sprintf('Training-set PPV–Recall curve (max F1 = %.2f%% at thr = %.4f)', ...
                      100*F1_best, thr_best));

        legend({'PPV vs Recall','Max-F1 operating point'}, ...
               'Location','southwest','Box','off');
    else
        warning('Insufficient class variation to build PPV–Recall curve.');
    end


%% === Helpers (PredLin PRESERVED EXACTLY) ===
function [T,Y]=PredLin(W,X)
%Predict with linear classifier
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
% Robust threshold chooser for A-score. No normalization. No perfcurve crash.
% - If both classes present: emulate PR max-F1 sweep manually
% - If only N in train:  threshold just above max score  -> predict no A
% - If only A in train:  threshold just below min score  -> predict all A
u = unique(yTrue);
if numel(u) < 2
    if all(yTrue==0)
        thr = max(aScore) + eps;   % predict none
    else
        thr = min(aScore) - eps;   % predict all
    end
    return
end

% Sweep unique score cutpoints
cuts = unique(aScore);
if numel(cuts) > 512
    % downsample cut grid for speed but keep coverage
    q = linspace(0,1,512);
    cuts = quantile(aScore, q);
    cuts = unique(cuts);
end

bestF1 = -inf; thr = median(aScore,'omitnan');
for c = cuts(:)'
    yhat = aScore >= c;
    TP = sum( yhat &  yTrue);
    FP = sum( yhat & ~yTrue);
    FN = sum(~yhat &  yTrue);
    if (2*TP + FP + FN) == 0, F1c = 0; else, F1c = (2*TP) / (2*TP + FP + FN); end
    if F1c > bestF1
        bestF1 = F1c; thr = c;
    end
end
end

% ---- local feature function (REPLACEMENT: flexible feature set) ----
function f = local_feats_flex(rr, rr_t, spo2_1hz, Fs, sampStart, sampEnd, FEATURES, fs_rr, lf_band, hf_band)
% rr      : RR intervals in seconds (length = numBeats-1), may contain NaN
% rr_t    : timestamp (in SAMPLES) of the 2nd beat for each RR
% spo2_1hz: 1 Hz SpO2 vector (double/NaN for bad seconds)
% Fs      : ECG sampling rate (200 Hz)
% sampStart/sampEnd: window bounds in SAMPLES (inclusive)
% FEATURES: cellstr list (order = column order)
% fs_rr   : resampling rate for RR(t) if freq features are used
% returns row vector f (1 × numel(FEATURES))

    % --- Slice RR for this window using midpoint timestamps (in samples) ---
    mask   = (rr_t >= sampStart) & (rr_t <= sampEnd);
    RRseg  = rr(mask);
    tRRwin = rr_t(mask);              % aligned timestamps for these RR intervals

    % Drop NaNs in RR **and** their timestamps to keep interp1 happy
    g      = isfinite(RRseg);
    RRseg  = RRseg(g);
    tRRwin = tRRwin(g);

    % --- Slice SpO2 in 1 Hz index space ---
    tStart = floor(sampStart/Fs) + 1;
    tEnd   = floor(sampEnd  /Fs);
    haveSp = (tStart >= 1) && (tEnd <= numel(spo2_1hz)) && (tStart <= tEnd);
    if haveSp
        sseg = spo2_1hz(tStart:tEnd);
    else
        sseg = [];
    end

    % Precompute dRR if we have enough beats
    if numel(RRseg) >= 2
        dRR   = diff(RRseg);
        absd  = abs(dRR);
    else
        dRR   = []; absd = [];
    end

    % Prepare frequency-domain RR(t) if requested and if we have enough data
    needFreq = any(ismember(FEATURES, {'LF','HF','LFHF','TotPow'}));
    haveFreq = false; LF=NaN; HF=NaN; LFHF=NaN; TotPow=NaN;

    if needFreq && numel(RRseg) >= 3
        % Build time axis for RR values (seconds) from cleaned window timestamps
        tRR  = (tRRwin / Fs);            % seconds
        tRR  = tRR(:);
        RRv  = RRseg(:);

        % Regularise to even grid [t0, t1] at fs_rr Hz
        t0   = max(min(tRR), sampStart/Fs);
        t1   = min(max(tRR), sampEnd  /Fs);
        if isfinite(t0) && isfinite(t1) && t1 > t0
            tgrid = (t0 : 1/fs_rr : t1).';
            if numel(tgrid) >= 16   % need enough points for PSD to be meaningful
                RR_even = interp1(tRR, RRv, tgrid, 'linear', 'extrap');
                RR_even = detrend(RR_even, 0);   % remove mean

                % Welch PSD
                nfft  = max(64, 2^nextpow2(numel(RR_even)));
                win   = min( round(0.5*fs_rr)*2, numel(RR_even)); % ~1 s-ish even at 4 Hz
                if win < 8, win = min(numel(RR_even), 8); end
                nover = floor(win/2);
                [Pxx, F] = pwelch(RR_even, win, nover, nfft, fs_rr);

                % Band powers
                LF     = bandpower(Pxx, F, lf_band,  'psd');
                HF     = bandpower(Pxx, F, hf_band,  'psd');
                TotPow = bandpower(Pxx, F, [lf_band(1) hf_band(2)], 'psd');
                if HF>0, LFHF = LF/HF; else, LFHF = NaN; end
                haveFreq = true;
            end
        end
    end

    % Build feature vector in the requested order
    f = nan(1, numel(FEATURES));
    for k = 1:numel(FEATURES)
        switch FEATURES{k}
            % ---- original (kept 1:1) ----
            case 'meanRR',     f(k) = mean(RRseg, 'omitnan');
            case 'stdRR',      f(k) = std( RRseg, 0, 'omitnan');
            case 'NN50',       f(k) = sum(absd > 0.050);
            case 'SDSD',       f(k) = std( dRR, 0, 'omitnan');
        
            % ---- SpO2 (robust, scalar) ----
            case 'meanSpO2'
                if ~isempty(sseg) && any(isfinite(sseg))
                    f(k) = mean(sseg, 'omitnan');
                else
                    f(k) = NaN;
                end
            case 'stdSpO2'
                if ~isempty(sseg) && any(isfinite(sseg))
                    f(k) = std(sseg, 0, 'omitnan');
                else
                    f(k) = NaN;
                end
            case 'minSpO2'
                if ~isempty(sseg) && any(isfinite(sseg))
                    tmp = min(sseg, [], 'omitnan');
                    if isempty(tmp), tmp = NaN; end
                    f(k) = tmp;
                else
                    f(k) = NaN;
                end
            case 'pctLT90'
                if ~isempty(sseg) && any(isfinite(sseg))
                    g = isfinite(sseg);
                    f(k) = mean(sseg(g) < 90) * 100;
                else
                    f(k) = NaN;
                end
            case 'spo2Slope'
                if ~isempty(sseg) && any(isfinite(sseg)) && numel(sseg) >= 3
                    x = (0:numel(sseg)-1)'; y = sseg(:);
                    g = isfinite(y);
                    if sum(g) >= 3
                        pfit = polyfit(x(g), y(g), 1);
                        f(k) = pfit(1);
                    else
                        f(k) = NaN;
                    end
                else
                    f(k) = NaN;
                end
        
            % ---- extra time-domain HRV ----
            case 'pNN50',      f(k) = sum(absd > 0.050) / max(1, numel(dRR));
            case 'RMSSD',      f(k) = sqrt(mean(dRR.^2, 'omitnan'));
            case 'medianRR',   f(k) = median(RRseg, 'omitnan');
            case 'iqrRR',      f(k) = iqr(RRseg);
            case 'logStdRR',   f(k) = log(std(RRseg,0,'omitnan') + 1e-6);
            case 'logSDSD',    f(k) = log(std(dRR,0,'omitnan') + 1e-6);
            case 'meanHR',     f(k) = 60 ./ mean(RRseg, 'omitnan');
        
            % ---- frequency-domain HRV ----
            case 'LF',         f(k) = haveFreq * LF     + (~haveFreq) * NaN;
            case 'HF',         f(k) = haveFreq * HF     + (~haveFreq) * NaN;
            case 'LFHF',       f(k) = haveFreq * LFHF   + (~haveFreq) * NaN;
            case 'TotPow',     f(k) = haveFreq * TotPow + (~haveFreq) * NaN;
        
            otherwise,         f(k) = NaN;
        end
    end

    % Guard: if we don't have at least 2 beats or valid SpO2 coverage, null out HRV/SpO2 fields
    if numel(RRseg) < 2
        % zero-out RR-based fields
        idx = ismember(FEATURES, {'meanRR','stdRR','NN50','SDSD','pNN50','RMSSD','medianRR','iqrRR','logStdRR','logSDSD','meanHR','LF','HF','LFHF','TotPow'});
        f(idx) = NaN;
    end
    if ~haveSp
        idx = ismember(FEATURES, {'meanSpO2','stdSpO2','minSpO2','pctLT90','spo2Slope'});
        f(idx) = NaN;
    end
end

% %% === LORO CV (NO normalisation): train-fit, train-threshold (robust), test-eval ===
% TP=0; FP=0; FN=0;
% TP_tr=0; FP_tr=0; FN_tr=0;   % aggregate XV training-set (in-sample) counts
% 
% for testIdx = 1:F
%     trIdx = setdiff(1:F, testIdx);
% 
%     Xtr = cat(1, X{trIdx});  Ttr = cat(1, T{trIdx});
%     Xte = X{testIdx};        Tte = T{testIdx};
% 
%     if isempty(Xtr) || isempty(Xte), continue; end
% 
%     % Fit linear regression on RAW train
%     W = TrainLR(Xtr, Ttr);                % outputs in [A N]
% 
%     % Choose threshold on TRAIN robustly (handles single-class cases)
%     [~, Ytr] = PredLin(W, Xtr);
%     aTr = Ytr(:,1); 
%     yTr = Ttr(:,1) > 0;
% 
%     thrA = choose_threshold_on_train(aTr, yTr);
% 
%     % --- XV TRAINING set evaluation (in-sample for this fold) ---
%     yhat_tr = aTr >= thrA;
%     TP_tr = TP_tr + sum( yhat_tr &  yTr);
%     FP_tr = FP_tr + sum( yhat_tr & ~yTr);
%     FN_tr = FN_tr + sum(~yhat_tr &  yTr);
% 
%     % Score TEST on RAW and apply threshold on A-score
%     [~, Yte] = PredLin(W, Xte);
%     aTe  = Yte(:,1); 
%     yTe  = Tte(:,1) > 0;
%     yHat = aTe >= thrA;
% 
%     % Collect outer-test scores/truth for PR
%     allScores{testIdx} = aTe(:);              % A-score from PredLin on TEST
%     allTruth{testIdx}  = logical(yTe(:));     % ground-truth A/N on TEST
% 
%     TP = TP + sum( yHat &  yTe);
%     FP = FP + sum( yHat & ~yTe);
%     FN = FN + sum(~yHat &  yTe);
% end
% 
% % === XV TRAINING set results (aggregated over folds) ===
% Sens_tr = 100 * TP_tr / max(1,TP_tr+FN_tr);
% PPV_tr  = 100 * TP_tr / max(1,TP_tr+FP_tr);
% F1_tr   = 100 * (2*TP_tr) / max(1,(2*TP_tr + FP_tr + FN_tr));
% fprintf('\n=== XV training set results ===\n');
% fprintf('F1  : %.2f%% | Sens: %.2f%% | PPV: %.2f%%\n', F1_tr, Sens_tr, PPV_tr);
% 
% % === XV TESTING set results (aggregated over folds) ===
% Sens = 100 * TP / max(1,TP+FN);
% PPV  = 100 * TP / max(1,TP+FP);
% F1   = 100 * (2*TP) / max(1,(2*TP + FP + FN));
% fprintf('\n=== XV testing set results ===\n');
% fprintf('F1  : %.2f%% | Sens: %.2f%% | PPV: %.2f%%\n', F1, Sens, PPV);
% % 
% % ===== Outer-test PR curve =====
% scores = vertcat(allScores{:});
% truth  = vertcat(allTruth{:});

% if ~isempty(scores)
%     [rec, prec, thr, ap, f1, iMax] = pr_curve_from_scores(scores, truth);
% 
%     figure; 
%     plot(rec, prec, 'LineWidth', 2); hold on;
%     plot(rec(iMax), prec(iMax), 'o', 'MarkerSize', 8, 'LineWidth', 2);
%     xlabel('Recall (Sensitivity)'); ylabel('Precision (PPV)');
%     title(sprintf('Outer-Test Precision–Recall  |  AUPRC = %.3f  |  maxF1 = %.3f', ap, f1(iMax)));
%     grid on; legend('PR curve','max-F1 point','Location','southwest');
% else
%     warning('No outer-test scores collected for PR curve.');
% end

% function [rec, prec, thr, ap, f1, iMax] = pr_curve_from_scores(scores, truth)
% % Build PR without perfcurve. scores = higher means more likely A.
% % truth must be logical vector (1 = A, 0 = N).
% 
%     scores = scores(:);
%     truth  = truth(:) > 0;
% 
%     % Unique cutpoints (descending for monotonic sweep)
%     thr = unique(scores(~isnan(scores)));
%     thr = sort(thr, 'descend');
% 
%     % Prepend + Inf and append - Inf to cover full range
%     thr = [inf; thr; -inf];
% 
%     P = zeros(numel(thr),1);
%     R = zeros(numel(thr),1);
%     F = zeros(numel(thr),1);
% 
%     pos = sum(truth);
%     for k = 1:numel(thr)
%         yhat = scores >= thr(k);
%         TP = sum( yhat &  truth);
%         FP = sum( yhat & ~truth);
%         FN = sum(~yhat &  truth);
% 
%         if TP+FP == 0, P(k) = 1; else, P(k) = TP/(TP+FP); end   % define P=1 at R=0
%         if TP+FN == 0, R(k) = 0; else, R(k) = TP/(TP+FN); end
%         if (P(k)+R(k))==0, F(k)=0; else, F(k)=2*P(k)*R(k)/(P(k)+R(k)); end
%     end
% 
%     % Ensure PR starts at (R=0,P=1) and is monotonic in recall for AP
%     [rec, ix] = sort(R, 'ascend');
%     prec = P(ix);
% 
%     % Average Precision: trapezoid on recall axis (simple AP approx)
%     ap = 0;
%     for i = 2:numel(rec)
%         ap = ap + (rec(i) - rec(i-1)) * ((prec(i) + prec(i-1))/2);
%     end
% 
%     f1 = F(ix);
%     [~, iMax] = max(f1);
% end
