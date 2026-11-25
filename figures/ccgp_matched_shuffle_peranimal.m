%% CCGP (rule, across pitch bands) vs MATCHED SHUFFLE, PER ANIMAL
%
% Re-runs the CCGP pipeline of ccgp_time_resolved_v3 (same features, same
% causal integration, same train/test resampling, same classifier, same
% rank-based AUC, same window averaging) and builds a matched null by running
% the IDENTICAL pipeline on label-shuffled data: ABCD/ABBA labels are shuffled
% within each pitch band (trial counts preserved), nNull times per session.
%
% CCGP directions (as in ccgp_time_resolved_v3):
%   0->1: train ABCD0 vs ABBA0, test ABCD1 vs ABBA1
%   1->0: train ABCD1 vs ABBA1, test ABCD0 vs ABBA0
%   CCGP = mean of the two directions (AUC; positive class = ABCD)
%
% Statistics, per window:
%   ACROSS ANIMALS (primary): per mouse, mean observed AUC minus the mean of
%   its matched shuffle distribution; tested across mice (one-sided t-test,
%   Wilcoxon, exact sign-flip). Individual mice need not be significant.
%   PER MOUSE: permutation P (mouse-mean observed vs mouse-mean of the k-th
%   shuffle across its sessions), Holm-corrected across mice.
%   For comparison: the animal-level test against 0.5 used in CCGP_peranimal.m.
%
% Defaults reproduce the call in CCGP_peranimal.m:
%   integrateWin 0.15 causal, smoothWin 0, diaglda, balanceTrain true,
%   balanceTest false, trainFrac 0.7, nResamples 10, baseline [-0.5 0).
%
% Requires: h5ephys_loader, Statistics and Machine Learning Toolbox.

clc; clear;

%% ------------------------------------------------------------------ settings
pth    = '/Users/athena/Work/Projects/Dammy/ephys analysis/';
h5file = fullfile(pth, 'unit_nozscore_file.h5');

opt = struct();
opt.dt     = 0.01;
opt.tStart = -0.5;
opt.tEnd   = 1.5;

% feature extraction (as ccgp_time_resolved_v3)
opt.integrateWin  = 0.15;
opt.integrateMode = "causal";       % "causal" or "centered"
opt.smoothWin     = 0.00;
opt.doBaseline    = true;
opt.baselineWindow = [-0.50 0.00];

% decoding (as ccgp_time_resolved_v3)
opt.classifier      = "diaglda";    % "diaglda" or "centroid"
opt.balanceTrain    = true;
opt.trainFrac       = 0.7;
opt.nResamples      = 10;
opt.minTrainPerCond = 5;
opt.minTestPerCond  = 3;
opt.sdFloor         = 1e-3;
opt.minTimeFrac     = 0.90;         % finite AUC in >= this fraction of analysed bins
opt.useAllTrials    = false;        % true: train on ALL trials of one band, test on ALL
                                    % of the other (valid for CCGP, since train and test
                                    % conditions are disjoint; more power). false = manuscript.

% selection (as ccgp_time_resolved_v3)
opt.nDiscard                 = 0;
opt.discardFirstGlobalTrials = 0;
opt.trialGlobalSuffix        = "_trialGlobal";

% windows (s) = summaryWindows of ccgp_time_resolved_v3; pips at 0,.25,.5,.75 (150 ms)
opt.windows  = [0.00 0.50;
                0.50 0.75;
                0.75 1.00;
                1.00 1.50];
opt.winNames = ["early (pips 1-2)", "pip3", "pip4", "post-sequence"];

% null
opt.nNull          = 200;               % shuffles per session (each runs the full pipeline)
opt.nResamplesNull = opt.nResamples;    % keep equal to nResamples for an exactly matched null
opt.seed  = 1;

% sessions per animal
opt.nSessPerAnimal = 20;        % Inf = all sessions; n = use at most n usable sessions per animal
opt.sessSelect     = "last";   % which n: "random" (seeded), "first" or "last" (DateSession order,
                                 % assumed chronological within animal)
opt.requireNSess   = false;      % true: drop animals with fewer than n usable sessions
opt.alpha = 0.05;

% optional: compare observed values with a saved ccgp_time_resolved_v3 output
opt.compareFile = "ccgp_time_resolved_v3_all.mat";

opt.outFile = "ccgp_matched_shuffle_peranimal_last20sess.mat";
opt.csvFile = "ccgp_matched_shuffle_peranimal_last20sess.csv";

%% ------------------------------------------------------------------ setup
idx = h5ephys_loader(h5file);
if ~isfield(idx, "h5file"), idx.h5file = h5file; end

t  = opt.tStart : opt.dt : opt.tEnd;
nT = numel(t);
idxB = find(t >= opt.baselineWindow(1) & t < opt.baselineWindow(2));
halfSmooth = max(0, round((opt.smoothWin/opt.dt)/2));

% integration indices per bin (as ccgp_time_resolved_v3)
intIdx = cell(1, nT);
fullBins = max(1, round(opt.integrateWin/opt.dt));
halfBins = max(1, round((opt.integrateWin/opt.dt)/2));
for ti = 1:nT
    if opt.integrateWin <= 0
        intIdx{ti} = ti;
    elseif opt.integrateMode == "centered"
        intIdx{ti} = max(1, ti-halfBins) : min(nT, ti+halfBins);
    else
        intIdx{ti} = max(1, ti-fullBins+1) : ti;
    end
end

% only the bins inside the windows are analysed
nW = size(opt.windows, 1);
binsUsed = find(t >= min(opt.windows(:)) & t < max(opt.windows(:)));
nB = numel(binsUsed);
winBins = cell(1, nW);
for w = 1:nW
    winBins{w} = find(t(binsUsed) >= opt.windows(w,1) & t(binsUsed) < opt.windows(w,2));
end
winLabels = arrayfun(@(k) sprintf('%.2f-%.2f s', opt.windows(k,1), opt.windows(k,2)), 1:nW, 'uni', 0);

% 1=ABCD0  2=ABBA0  3=ABCD1  4=ABBA1
cond = table(["A-0";"A-1";"A-2";"A-3"], ["ABCD0";"ABBA0";"ABCD1";"ABBA1"], ...
             'VariableNames', {'AFile','Pattern'});

T = idx.T;
if opt.nDiscard > 0
    keepSessions = strings(0,1);
    animals = string(T.Animal);
    for a = unique(animals)'
        dsList = sort(unique(T.DateSession(animals==a)));
        nDrop  = min(opt.nDiscard, numel(dsList));
        keepSessions = [keepSessions; dsList(nDrop+1:end)]; %#ok<AGROW>
    end
    T = T(ismember(T.DateSession, keepSessions), :);
end
dateSessions = unique(T.DateSession);
nS = numel(dateSessions);

obsW  = nan(nS, nW);
nullW = nan(nS, nW, opt.nNull);
sessAnimal = strings(nS,1);
valid = false(nS,1);

rsObs  = RandStream('mt19937ar', 'Seed', opt.seed);
rsNull = RandStream('mt19937ar', 'Seed', opt.seed + 1);

%% ------------------------------------------------------------------ per session
tic
% session order: with nSessPerAnimal = Inf this is 1:nS (identical to earlier runs);
% otherwise sessions are visited per animal in the chosen order and each animal
% stops after n USABLE sessions (sessions failing inclusion are replaced by the next)
procOrder = session_order(dateSessions, opt);
sessAnimalAll = extractBefore(string(dateSessions), "_");
animList = unique(sessAnimalAll);
nUsedPerAnimal = zeros(numel(animList), 1);
for ii = 1:nS
    s  = procOrder(ii);
    ai = find(animList == sessAnimalAll(s));
    if nUsedPerAnimal(ai) >= opt.nSessPerAnimal, continue; end
    ds = dateSessions(s);
    Ts = T(T.DateSession == ds, :);
    sessAnimal(s) = extractBefore(string(ds), "_");

    Fc = cell(1,4);  ok = true;
    for c = 1:4
        rr = Ts(Ts.AFile == cond.AFile(c), :);
        if isempty(rr), ok = false; break; end
        APath = string(rr.APath(1));
        X = double(idx.read(APath));                       % [time x units x trials]
        if size(X,1) ~= nT, error("Time mismatch in %s", APath); end
        if opt.discardFirstGlobalTrials > 0
            g = double(h5read(idx.h5file, APath + string(opt.trialGlobalSuffix)));
            keep = g(:) > opt.discardFirstGlobalTrials;
            if sum(keep) < opt.minTrainPerCond + opt.minTestPerCond, ok = false; break; end
            X = X(:,:,keep);
        end
        if opt.doBaseline, X = X - mean(X(idxB,:,:), 1); end
        if halfSmooth > 0, X = movmean(X, [halfSmooth halfSmooth], 1, "Endpoints", "shrink"); end

        % integrated features for the analysed bins: [bins x units x trials]
        F = zeros(nB, size(X,2), size(X,3));
        for b = 1:nB
            F(b,:,:) = mean(X(intIdx{binsUsed(b)},:,:), 1);
        end
        Fc{c} = F;
    end
    if ~ok, continue; end

    nTr = cellfun(@(F) size(F,3), Fc);
    nTrain = max(opt.minTrainPerCond, floor(opt.trainFrac * nTr));
    if ~opt.useAllTrials && any(nTrain < opt.minTrainPerCond | (nTr - nTrain) < opt.minTestPerCond)
        continue
    end

    % observed
    aucObs = ccgp_session(Fc, opt.nResamples, opt, rsObs);
    if mean(isfinite(aucObs)) < opt.minTimeFrac, continue; end
    for w = 1:nW, obsW(s,w) = mean(aucObs(winBins{w}), 'omitnan'); end

    % matched null: same pipeline, rule labels shuffled within each band
    for k = 1:opt.nNull
        Fs = shuffle_within_band(Fc, rsNull);
        aucNull = ccgp_session(Fs, opt.nResamplesNull, opt, rsNull);
        for w = 1:nW, nullW(s,w,k) = mean(aucNull(winBins{w}), 'omitnan'); end
    end

    valid(s) = true;
    nUsedPerAnimal(ai) = nUsedPerAnimal(ai) + 1;
    fprintf('Session %3d/%d  %-20s  units=%3d  trials=[%s]  CCGP(post-seq)=%.3f  null=%.3f  (%.0f s)\n', ...
        s, nS, ds, size(Fc{1},2), num2str(nTr), obsW(s,nW), mean(nullW(s,nW,:),'omitnan'), toc);
end

valid = apply_min_sessions(valid, sessAnimalAll, animList, nUsedPerAnimal, opt);
v = find(valid);
if isempty(v), error('No valid sessions.'); end
animal = sessAnimal(v);
[G, mice] = findgroups(animal);
nMice = numel(mice);

%% ------------------------------------------------------------------ optional sanity check
if strlength(opt.compareFile) > 0 && isfile(opt.compareFile)
    L = load(opt.compareFile, "Out");  O = L.Out;
    [tf, loc] = ismember(string(dateSessions(v)), string(O.sessionIDs));
    fprintf('\nSanity check vs %s (%d matched sessions):\n', opt.compareFile, sum(tf));
    for w = 1:nW
        ww = O.t >= opt.windows(w,1) & O.t < opt.windows(w,2);
        theirs = mean(O.auc_session(loc(tf), ww), 2, 'omitnan');
        ours   = obsW(v(tf), w);
        fprintf('   %-18s r = %.3f, mean diff = %+.4f\n', opt.winNames(w), corr(theirs, ours, 'rows', 'complete'), ...
            mean(ours - theirs, 'omitnan'));
    end
    fprintf('   (differences come only from random train/test splits; r should be high)\n');
end

%% ------------------------------------------------------------------ statistics
rows = struct([]);
summary = struct([]);
flips = 1 - 2*(dec2bin(0:2^nMice-1, nMice) - '0');           % all sign patterns

for w = 1:nW
    O  = obsW(v, w);
    Nl = reshape(nullW(v, w, :), numel(v), []);              % sessions x nNull

    mouseObs = nan(nMice,1);  mouseNull = nan(nMice, opt.nNull);
    for m = 1:nMice
        ss = (G == m);
        mouseObs(m)    = mean(O(ss), 'omitnan');
        mouseNull(m,:) = mean(Nl(ss,:), 1, 'omitnan');
        r = struct();
        r.window = string(winLabels{w});  r.windowName = opt.winNames(w);
        r.mouse = mice(m);  r.nSessions = sum(ss);
        r.ccgp = mouseObs(m);  r.nullMean = mean(mouseNull(m,:));
        r.null05 = prctile(mouseNull(m,:), 5);  r.null95 = prctile(mouseNull(m,:), 95);
        r.diff = mouseObs(m) - r.nullMean;
        r.pPerm = (1 + sum(mouseNull(m,:) >= mouseObs(m))) / (1 + opt.nNull);
        r.pPermHolm = NaN;
        if isempty(rows), rows = r; else, rows(end+1,1) = r; end %#ok<AGROW>
    end
    ii = find(arrayfun(@(x) x.window == string(winLabels{w}), rows));
    pa = holm([rows(ii).pPerm]);
    for j = 1:numel(ii), rows(ii(j)).pPermHolm = pa(j); end

    % across animals: observed - matched shuffle mean
    dM = mouseObs - mean(mouseNull, 2);
    [~, pT, ~, st] = ttest(dM, 0, 'Tail', 'right');
    [~, ~, ci]     = ttest(dM, 0);
    pWil  = signrank(dM, 0, 'tail', 'right');
    pFlip = mean(mean(flips .* dM.', 2) >= mean(dM) - 1e-12);

    % for comparison: animal-level test against 0.5 (as CCGP_peranimal.m)
    [~, p05, ~, st05] = ttest(mouseObs, 0.5, 'Tail', 'right');

    % pooled permutation (fixed effects)
    pPooled = (1 + sum(mean(mouseNull,1) >= mean(mouseObs))) / (1 + opt.nNull);

    S = struct('window', winLabels{w}, 'windowName', opt.winNames(w), ...
        'ccgpPerMouse', mouseObs, 'nullMeanPerMouse', mean(mouseNull,2), 'diffPerMouse', dM, ...
        'diffMean', mean(dM), 'diffCI95', ci, 'dz', mean(dM)/std(dM), ...
        't', st.tstat, 'df', st.df, 'pT', pT, 'pWilcoxon', pWil, 'pSignFlip', pFlip, ...
        't_vs05', st05.tstat, 'p_vs05', p05, 'pPooled', pPooled, ...
        'nMiceSig', sum([rows(ii).pPerm] < opt.alpha), 'nMiceSigHolm', sum(pa < opt.alpha));
    if isempty(summary), summary = S; else, summary(end+1) = S; end %#ok<AGROW>
end
R = struct2table(rows);

%% ------------------------------------------------------------------ print
fprintf('\n=============== CCGP vs MATCHED SHUFFLE (AUC, rule across pitch bands) ===============\n');
fprintf('%d sessions from %d mice | %d shuffles per session | classifier %s | integrate %.0f ms %s | all tests one-sided\n', ...
    numel(v), nMice, opt.nNull, opt.classifier, 1000*opt.integrateWin, opt.integrateMode);
for w = 1:nW
    S = summary(w);
    fprintf('\n--- %s (%s) ---\n', S.window, S.windowName);
    fprintf('ACROSS MICE, CCGP - matched shuffle mean: %+.4f, 95%% CI [%.4f, %.4f], dz = %.2f\n', ...
        S.diffMean, S.diffCI95(1), S.diffCI95(2), S.dz);
    fprintf('   t(%d) = %.2f, P = %.4f | Wilcoxon P = %.4f | exact sign-flip P = %.4f (min possible %.4f)\n', ...
        S.df, S.t, S.pT, S.pWilcoxon, S.pSignFlip, 1/2^nMice);
    fprintf('   for comparison, mouse means vs 0.5: t(%d) = %.2f, P = %.4f\n', S.df, S.t_vs05, S.p_vs05);
    fprintf('   pooled permutation (fixed effects): P = %.4f | mice significant: %d/%d uncorrected, %d/%d Holm\n', ...
        S.pPooled, S.nMiceSig, nMice, S.nMiceSigHolm, nMice);
    fprintf('   %-8s %4s %8s %8s %8s %8s %8s\n', 'mouse', 'n', 'CCGP', 'null', 'diff', 'Pperm', 'Pholm');
    Rk = R(R.window == string(winLabels{w}), :);
    for j = 1:height(Rk)
        fprintf('   %-8s %4d %8.4f %8.4f %+8.4f %8.4f %8.4f%s\n', Rk.mouse(j), Rk.nSessions(j), Rk.ccgp(j), ...
            Rk.nullMean(j), Rk.diff(j), Rk.pPerm(j), Rk.pPermHolm(j), repmat(' *', 1, Rk.pPermHolm(j) < opt.alpha));
    end
end

save(opt.outFile, "R", "summary", "opt", "obsW", "nullW", "v", "animal", "mice", "dateSessions");
writetable(R, opt.csvFile);
fprintf('\nSaved %s and %s\n', opt.outFile, opt.csvFile);

%% ------------------------------------------------------------------ figure
load("ccgp_matched_shuffle_peranimal_last20sess.mat");
nMice = numel(mice);
nW    = size(obsW, 2);
nNull = size(nullW, 3);
winLabels = arrayfun(@(k) sprintf('%.2f-%.2f s', opt.windows(k,1), opt.windows(k,2)), 1:nW, 'uni', 0);
flips = 1 - 2*(dec2bin(0:2^nMice-1, nMice) - '0');     % all sign patterns across mice


figure('Name','CCGP vs matched shuffle','Color','w','Position',[80 80 260*nW 320]);
for w = 1:nW
    subplot(1, nW, w); hold on
    Rk = R(R.window == string(winLabels{w}), :);
    for j = 1:height(Rk)
        plot([j j], [Rk.null05(j) Rk.null95(j)], '-', 'Color', [0.75 0.75 0.75], 'LineWidth', 6);
        plot(j, Rk.ccgp(j), 'ko', 'MarkerFaceColor', 'k');
    end
    yline(0.5, 'k:');
    ylim([0.436 0.55])
    set(gca, 'XTick', 1:height(Rk), 'XTickLabel', Rk.mouse, 'TickDir', 'out', 'Box', 'off', ...
        'XLim', [0.5 height(Rk)+0.5]);
    if w == 1, ylabel('CCGP (AUC)'); end
    title(sprintf('%s\n%s', winLabels{w}, opt.winNames(w)), 'FontWeight', 'normal');
end
% grey: 5th-95th percentile of each mouse's matched null; dot: observed mouse mean

%% ================================================================== helpers
function auc = ccgp_session(Fc, nRes, opt, rs)
% Fc: 1x4 cell {ABCD0, ABBA0, ABCD1, ABBA1}, each [bins x units x trials]
nB = size(Fc{1}, 1);
A  = nan(nB, nRes);
for r = 1:nRes
    tr = cell(1,4);  te = cell(1,4);
    for c = 1:4
        n = size(Fc{c}, 3);
        if opt.useAllTrials
            tr{c} = 1:n;  te{c} = 1:n;
        else
            nTrain = max(opt.minTrainPerCond, floor(opt.trainFrac * n));
            p = randperm(rs, n);
            tr{c} = p(1:nTrain);  te{c} = p(nTrain+1:end);
        end
    end
    a01 = dir_auc(Fc{1}(:,:,tr{1}), Fc{2}(:,:,tr{2}), Fc{3}(:,:,te{3}), Fc{4}(:,:,te{4}), opt, rs);
    a10 = dir_auc(Fc{3}(:,:,tr{3}), Fc{4}(:,:,tr{4}), Fc{1}(:,:,te{1}), Fc{2}(:,:,te{2}), opt, rs);
    A(:,r) = 0.5 * (a01 + a10);
end
auc = mean(A, 2, 'omitnan');
end

function auc = dir_auc(Ptr, Ntr, Pte, Nte, opt, rs)
% train on Ptr (ABCD) vs Ntr (ABBA); score Pte/Nte; AUC per bin
nB = size(Ptr, 1);
if opt.balanceTrain
    n = min(size(Ptr,3), size(Ntr,3));
    Ptr = Ptr(:,:, randperm(rs, size(Ptr,3), n));
    Ntr = Ntr(:,:, randperm(rs, size(Ntr,3), n));
end
muP = mean(Ptr, 3);  muN = mean(Ntr, 3);                 % bins x units
switch opt.classifier
    case "centroid"
        w = muP - muN;
    case "diaglda"
        vv = max(var(cat(3, Ptr, Ntr), 0, 3), opt.sdFloor^2);   % as ccgp_time_resolved_v3
        w = (muP - muN) ./ vv;
    otherwise
        error('classifier must be "centroid" or "diaglda".');
end
b  = -0.5 * sum((muP + muN) .* w, 2);                    % bins x 1
sP = reshape(sum(Pte .* w, 2), nB, []) + b;              % bins x nPosTest
sN = reshape(sum(Nte .* w, 2), nB, []) + b;              % bins x nNegTest
if size(sP,2) < 2 || size(sN,2) < 2, auc = nan(nB,1); return; end
D   = sP - permute(sN, [1 3 2]);                          % bins x nP x nN
auc = mean(mean((D > 0) + 0.5*(D == 0), 3), 2);          % Mann-Whitney AUC (ties = 0.5)
end

function Fs = shuffle_within_band(Fc, rs)
Fs = Fc;
for pr = [1 2; 3 4]'                                       % band 0: {1,2}; band 1: {3,4}
    P  = cat(3, Fc{pr(1)}, Fc{pr(2)});
    i  = randperm(rs, size(P,3));
    n1 = size(Fc{pr(1)}, 3);
    Fs{pr(1)} = P(:,:, i(1:n1));
    Fs{pr(2)} = P(:,:, i(n1+1:end));
end
end

function padj = holm(p)
p = p(:);  m = numel(p);
[ps, ix] = sort(p);
adj = min(1, cummax((m - (1:m)' + 1) .* ps));
padj = nan(m,1);  padj(ix) = adj;
end

%% ================================================================== session selection
function order = session_order(dateSessions, opt)
nS = numel(dateSessions);
if ~isfinite(opt.nSessPerAnimal), order = (1:nS)'; return; end
anim = extractBefore(string(dateSessions), "_");
rs = RandStream('mt19937ar', 'Seed', opt.seed + 20);
order = zeros(0,1);
for a = unique(anim)'
    ii = find(anim == a);                        % dateSessions are sorted: chronological within animal
    switch opt.sessSelect
        case "random", ii = ii(randperm(rs, numel(ii)));
        case "first"   % keep order
        case "last",   ii = flipud(ii(:));
        otherwise, error('opt.sessSelect must be "random", "first" or "last".');
    end
    order = [order; ii(:)]; %#ok<AGROW>
end
end

function valid = apply_min_sessions(valid, sessAnimalAll, animList, nUsed, opt)
if isfinite(opt.nSessPerAnimal)
    fprintf('\nSessions per animal (cap %d, %s): %s\n', opt.nSessPerAnimal, opt.sessSelect, ...
        strjoin(animList(:)' + "=" + string(nUsed(:)'), ', '));
    if opt.requireNSess
        for k = find(nUsed(:)' < opt.nSessPerAnimal)
            valid(sessAnimalAll == animList(k)) = false;
            fprintf('   dropping %s: only %d usable sessions\n', animList(k), nUsed(k));
        end
    end
end
end
