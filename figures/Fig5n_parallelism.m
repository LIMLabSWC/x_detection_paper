clc
clear all

pth = '/Users/athena/Work/Projects/Dammy/ephys analysis/';
h5file = [pth,'/unit_nozscore_file.h5'];
idx = h5ephys_loader(h5file);

idx.animals          % list all animals
idx.summary          % how many A-* datasets per animal

Out = lda_axes_statespace_cv50_rect_v3(idx, "all", ...
  "axisFitNSplits", 5, ...
  "integrateWinEval", 0.15, ...
  "integrateMode", "causal", ...
  "pitchTrainWindow", [0.0 0.5], ...
  "ruleTrainWindow",  [0.7 1.1], ...
  "rectWindow", [0.75 1.0], ...
  "rectWindowPitch", [0.15 0.5], ...  % x centroid window
  "rectWindowRule",  [0.70 1.1], ...  % y centroid window
  "leakageMode", "signed", ...
  "allowAntiparallel", true, ...
  "pitchSummaryWindow",[0.2 0.4], ...
  "ruleSummaryWindow",[0.70 1.10], ...
  "plot", true);
save("lda_axes_statespace_cv50_rect_Pitch0to500_Rule750to1_Out_test.mat", "Out")
load("lda_axes_statespace_cv50_rect_Pitch0to500_Rule750to1_Out_test.mat")
% 
plot_lda_geometry_from_out(Out);


function Out = lda_axes_statespace_cv50_rect_v5(idx, animalID, varargin)
%LDA_AXES_STATESPACE_CV50_RECT_V5
% Adds TIME-RESOLVED LDA projection for each of the 4 sequences.
%
% At every time bin (width = integrateWinEval, stepped by dt), the
% population response is projected onto the Pitch and Rule LDA axes,
% which were trained on pitchTrainWindow / ruleTrainWindow respectively.
%
% Two display modes (projectionMode):
%   "normalized"     - raw projection scaled so that at the training
%                      window the class-0 mean -> -1 and class-1 mean -> +1,
%                      centred at 0. The same scale is applied to every
%                      time bin, so values outside [-1,1] are possible.
%                        Pitch axis: class-0 = ABCD0/ABBA0  class-1 = ABCD1/ABBA1
%                        Rule  axis: class-0 = ABCD0/ABCD1  class-1 = ABBA0/ABBA1
%   "percent_class1" - fraction of test trials per condition with
%                      projection > 0 (classified as class 1), as %.
%
% New parameters vs v3/v4:
%   projectionMode          "normalized" | "percent_class1"  (default: "normalized")
%   plotProjection          true | false                     (default: true)
%   projectionWhich         "test" | "train" | "both"        (default: "test")
%   projectionShowSessions  true | false  - overlay per-session thin lines
%   projectionShowPips      true | false  - mark pip onset times
%
% All v3 parameters are preserved unchanged.

p = inputParser;

% ---- time axis ----
p.addParameter("dt",     0.01);
p.addParameter("tStart", -0.5);
p.addParameter("tEnd",   1.5);

% ---- windows (seconds) ----
p.addParameter("baselineWindow",   [-0.50  0.00]);
p.addParameter("pitchTrainWindow", [ 0.00  0.50]);
p.addParameter("ruleTrainWindow",  [ 0.55  1.05]);

% Geometry windows (kept from v3)
p.addParameter("rectWindow",      [0.8 1.0]);
p.addParameter("rectWindowPitch", []);
p.addParameter("rectWindowRule",  []);

% ---- eval integration ----
p.addParameter("integrateWinEval", 0.10);
p.addParameter("integrateMode",    "causal");

% ---- preprocessing ----
p.addParameter("doBaseline",    true);
p.addParameter("zscoreNeurons", true);
p.addParameter("sdFloor",       1e-3);
p.addParameter("minUnits",      10);

% ---- LDA ----
p.addParameter("ldaMode",      "ridge");
p.addParameter("lambda",       0.10);
p.addParameter("balanceTrain", true);

% ---- CV ----
p.addParameter("axisFitFrac",    0.5);
p.addParameter("axisFitNSplits", 10);
p.addParameter("axisFitSeed",    1);

% ---- filtering ----
p.addParameter("nDiscardSessions",         0);
p.addParameter("discardFirstGlobalTrials", 0);
p.addParameter("trialGlobalSuffix",        "_trialGlobal");

% ---- leakage options ----
p.addParameter("leakageMode",        "unsigned");
p.addParameter("pitchSummaryWindow", [0.05 0.45]);
p.addParameter("ruleSummaryWindow",  [0.80 1.20]);
p.addParameter("minOnAxisDelta",     0.005);
p.addParameter("ratioEps",           1e-6);
p.addParameter("allowAntiparallel",  true);

% ---- plotting (legacy) ----
p.addParameter("plot",               true);
p.addParameter("forceFigureVisible", true);
p.addParameter("showPips",           true);
p.addParameter("plotAUC",            true);
p.addParameter("plotRectangle",      true);
p.addParameter("plotEllipses",       true);
p.addParameter("plotLeakageSummary", true);
p.addParameter("plotTrajectories",   true);
p.addParameter("trajectoryWhich",    "test");
p.addParameter("trajectoryShowSessionSpaghetti", false);
p.addParameter("trajectoryShowSessionMeans",     false);
p.addParameter("trajectoryShowGrandMean",        true);
p.addParameter("trajectoryMarkPipStarts",        true);
p.addParameter("trajectoryEqualAxis",            false);

% ---- NEW: time-resolved projection ----
p.addParameter("plotProjection",         true);
p.addParameter("projectionMode",         "normalized");
p.addParameter("projectionWhich",        "test");
p.addParameter("projectionShowSessions", false);
p.addParameter("projectionShowPips",     true);

p.parse(varargin{:});
opt = p.Results;

% ---- normalise string options ----
if iscell(animalID), animalID = string(animalID); end
animalID = string(animalID);
opt.integrateMode   = lower(string(opt.integrateMode));
opt.ldaMode         = lower(string(opt.ldaMode));
opt.leakageMode     = lower(string(opt.leakageMode));
opt.trajectoryWhich = lower(string(opt.trajectoryWhich));
opt.projectionMode  = lower(string(opt.projectionMode));
opt.projectionWhich = lower(string(opt.projectionWhich));

assert(any(opt.leakageMode     == ["signed","unsigned"]),           'leakageMode must be "signed" or "unsigned".');
assert(any(opt.trajectoryWhich == ["test","all"]),                  'trajectoryWhich must be "test" or "all".');
assert(any(opt.projectionMode  == ["normalized","percent_class1"]), 'projectionMode must be "normalized" or "percent_class1".');
assert(any(opt.projectionWhich == ["test","train","both"]),         'projectionWhich must be "test", "train", or "both".');

% ---- rect windows ----
if isempty(opt.rectWindowPitch), opt.rectWindowPitch = opt.rectWindow; end
if isempty(opt.rectWindowRule),  opt.rectWindowRule  = opt.rectWindow; end
opt.rectWindowPitch = double(opt.rectWindowPitch(:)).';
opt.rectWindowRule  = double(opt.rectWindowRule(:)).';

% ---- time vector and index sets ----
t  = opt.tStart : opt.dt : opt.tEnd;
nT = numel(t);

idxB         = find(t >= opt.baselineWindow(1)   & t < opt.baselineWindow(2));
idxPitch     = find(t >= opt.pitchTrainWindow(1) & t < opt.pitchTrainWindow(2));
idxRule      = find(t >= opt.ruleTrainWindow(1)  & t < opt.ruleTrainWindow(2));
idxRectPitch = find(t >= opt.rectWindowPitch(1)  & t < opt.rectWindowPitch(2));
idxRectRule  = find(t >= opt.rectWindowRule(1)   & t < opt.rectWindowRule(2));

if isempty(idxRectPitch), warning("rectWindowPitch is empty!"); end
if isempty(idxRectRule),  warning("rectWindowRule is empty!");  end

evalTimeIdx = make_time_windows(t, opt.integrateWinEval, opt.integrateMode);

% ---- condition map ----
% 1=ABCD0  2=ABBA0  3=ABCD1  4=ABBA1
% Pitch class: 0={1,2}  1={3,4}
% Rule  class: 0={1,3}  1={2,4}
cond = table( ...
    ["A-0";"A-1";"A-2";"A-3"], ...
    ["ABCD0";"ABBA0";"ABCD1";"ABBA1"], ...
    'VariableNames',{'AFile','Pattern'});
patterns = string(cond.Pattern);

% ---- session selection ----
T = idx.T;
if isscalar(animalID) && animalID=="all"
    % keep all
elseif isscalar(animalID)
    T = T(T.Animal == animalID, :);
else
    T = T(ismember(T.Animal, animalID), :);
end

if opt.nDiscardSessions > 0
    keepSessions = strings(0,1);
    for a = unique(T.Animal)'
        dsList = sort(unique(T.DateSession(T.Animal==a)));
        nDrop  = min(opt.nDiscardSessions, numel(dsList));
        keepSessions = [keepSessions; dsList(nDrop+1:end)];
    end
    T = T(ismember(T.DateSession, keepSessions), :);
end
dateSessions = unique(T.DateSession);

% ---- storage ----
nS = numel(dateSessions);
valid      = false(nS,1);
sessAnimal = strings(nS,1);
nUnitsSess = nan(nS,1);

aucPP = nan(nS,nT);  aucRR = nan(nS,nT);
aucPR = nan(nS,nT);  aucRP = nan(nS,nT);

pts_rect_test  = nan(nS,4,2);
pts_rect_train = nan(nS,4,2);
rect_score     = nan(nS,1);
pitch_par_cos  = nan(nS,1);  rule_par_cos  = nan(nS,1);
pitch_par_abs  = nan(nS,1);  rule_par_abs  = nan(nS,1);

traj2D = nan(nS,4,nT,2);

% NEW: time-resolved projection  [nS x 4 x nT]
proj_norm_pitch_test  = nan(nS,4,nT);
proj_norm_pitch_train = nan(nS,4,nT);
proj_norm_rule_test   = nan(nS,4,nT);
proj_norm_rule_train  = nan(nS,4,nT);

pct1_pitch_test  = nan(nS,4,nT);
pct1_pitch_train = nan(nS,4,nT);
pct1_rule_test   = nan(nS,4,nT);
pct1_rule_train  = nan(nS,4,nT);

% ================================================================
%  MAIN SESSION LOOP
% ================================================================
for s = 1:nS
    ds = dateSessions(s);
    Ts = T(T.DateSession==ds,:);
    sessAnimal(s) = extractBefore(string(ds),"_");

    Xc = cell(4,1);  ok = true;
    for c = 1:4
        rr = Ts(Ts.AFile==cond.AFile(c),:);
        if isempty(rr), ok=false; break; end
        APath = string(rr.APath(1));
        X = double(idx.read(APath));
        if size(X,1) ~= nT, error("Time mismatch"); end

        if opt.discardFirstGlobalTrials > 0
            if ~isfield(idx,"h5file"), error("discardFirstGlobalTrials requires idx.h5file"); end
            g = double(h5read(idx.h5file, APath + string(opt.trialGlobalSuffix)));
            keep = g(:) > opt.discardFirstGlobalTrials;
            if sum(keep) < 4, ok=false; break; end
            X = X(:,:,keep);
        end
        if opt.doBaseline
            X = X - mean(X(idxB,:,:),1);
        end
        Xc{c} = X;
    end
    if ~ok, continue; end

    nUnits = size(Xc{1},2);
    if nUnits < opt.minUnits, continue; end
    nUnitsSess(s) = nUnits;

    % per-split accumulators
    aucPP_r = nan(opt.axisFitNSplits,nT);
    aucRR_r = nan(opt.axisFitNSplits,nT);
    aucPR_r = nan(opt.axisFitNSplits,nT);
    aucRP_r = nan(opt.axisFitNSplits,nT);

    pts_r_test  = nan(opt.axisFitNSplits,4,2);
    pts_r_train = nan(opt.axisFitNSplits,4,2);
    traj2D_r    = nan(opt.axisFitNSplits,4,nT,2);

    % NEW: [nSplits x 4 x nT]
    pnP_test_r  = nan(opt.axisFitNSplits,4,nT);
    pnP_train_r = nan(opt.axisFitNSplits,4,nT);
    pnR_test_r  = nan(opt.axisFitNSplits,4,nT);
    pnR_train_r = nan(opt.axisFitNSplits,4,nT);

    p1P_test_r  = nan(opt.axisFitNSplits,4,nT);
    p1P_train_r = nan(opt.axisFitNSplits,4,nT);
    p1R_test_r  = nan(opt.axisFitNSplits,4,nT);
    p1R_train_r = nan(opt.axisFitNSplits,4,nT);

    % ============================================================
    %  CV SPLIT LOOP
    % ============================================================
    for r = 1:opt.axisFitNSplits
        rng(opt.axisFitSeed + 1000*s + r);

        fitIdx  = cell(4,1);  testIdx = cell(4,1);
        for c = 1:4
            nTr  = size(Xc{c},3);
            nFit = max(2, floor(opt.axisFitFrac * nTr));
            perm = randperm(nTr);
            fitIdx{c}  = perm(1:nFit);
            testIdx{c} = perm(nFit+1:end);
            if numel(testIdx{c}) < 1, testIdx{c} = fitIdx{c}; end
        end

        % ---- fit features (train trials, training windows) ----
        Xlow_fit  = [feat_trials(Xc{1},idxPitch,fitIdx{1}); feat_trials(Xc{2},idxPitch,fitIdx{2})];
        Xhigh_fit = [feat_trials(Xc{3},idxPitch,fitIdx{3}); feat_trials(Xc{4},idxPitch,fitIdx{4})];
        yPitch_fit = [zeros(size(Xlow_fit,1),1); ones(size(Xhigh_fit,1),1)];
        XPitch_fit = [Xlow_fit; Xhigh_fit];

        Xabcd_fit = [feat_trials(Xc{1},idxRule,fitIdx{1}); feat_trials(Xc{3},idxRule,fitIdx{3})];
        Xabba_fit = [feat_trials(Xc{2},idxRule,fitIdx{2}); feat_trials(Xc{4},idxRule,fitIdx{4})];
        yRule_fit  = [zeros(size(Xabcd_fit,1),1); ones(size(Xabba_fit,1),1)];
        XRule_fit  = [Xabcd_fit; Xabba_fit];

        % ---- z-score ----
        if opt.zscoreNeurons
            Xpool = [XPitch_fit; XRule_fit];
            muZ = mean(Xpool,1);  sdZ = max(std(Xpool,0,1), opt.sdFloor);
            XPitch_fit = (XPitch_fit - muZ) ./ sdZ;
            XRule_fit  = (XRule_fit  - muZ) ./ sdZ;
        else
            muZ = zeros(1,nUnits);  sdZ = ones(1,nUnits);
        end

        % ---- balance and fit axes ----
        if opt.balanceTrain
            [XPb,yPb] = balance_binary(XPitch_fit, yPitch_fit);
            [XRb,yRb] = balance_binary(XRule_fit,  yRule_fit);
        else
            XPb=XPitch_fit; yPb=yPitch_fit; XRb=XRule_fit; yRb=yRule_fit;
        end

        wP = lda_axis(XPb, yPb, opt.ldaMode, opt.lambda, opt.sdFloor);
        wR = lda_axis(XRb, yRb, opt.ldaMode, opt.lambda, opt.sdFloor);
        wP = wP / max(norm(wP),eps);
        wR = wR / max(norm(wR),eps);

        % sign alignment: class 1 -> positive projection
        scP = XPitch_fit*wP;  if mean(scP(yPitch_fit==1)) < mean(scP(yPitch_fit==0)), wP=-wP; end
        scR = XRule_fit *wR;  if mean(scR(yRule_fit ==1)) < mean(scR(yRule_fit ==0)), wR=-wR; end

        % ============================================================
        %  TIME-RESOLVED LOOP: AUC + trajectories + projections
        % ============================================================

        % Accumulate raw mean projections [4 x nT] over time bins
        rawP_test  = nan(4,nT);   rawP_train = nan(4,nT);
        rawR_test  = nan(4,nT);   rawR_train = nan(4,nT);

        for ti = 1:nT
            tw = evalTimeIdx{ti};

            % z-scored features at this time window for all conditions
            F      = cell(4,1);
            Ftrain = cell(4,1);
            for c = 1:4
                F{c}      = (feat_trials(Xc{c}, tw, testIdx{c}) - muZ) ./ sdZ;
                Ftrain{c} = (feat_trials(Xc{c}, tw, fitIdx{c})  - muZ) ./ sdZ;
            end

            % ---- AUC (unchanged from v3) ----
            Xp = [F{1};F{2};F{3};F{4}];
            yp = [zeros(size(F{1},1)+size(F{2},1),1); ones(size(F{3},1)+size(F{4},1),1)];
            Xr = [F{1};F{3};F{2};F{4}];
            yr = [zeros(size(F{1},1)+size(F{3},1),1); ones(size(F{2},1)+size(F{4},1),1)];

            aucPP_r(r,ti) = auc_rank(yp, Xp*wP);
            aucRR_r(r,ti) = auc_rank(yr, Xr*wR);
            aucPR_r(r,ti) = auc_rank(yp, Xp*wR);
            aucRP_r(r,ti) = auc_rank(yr, Xr*wP);

            % ---- 2D trajectory ----
            for c = 1:4
                if opt.trajectoryWhich=="all", trUse=1:size(Xc{c},3); else, trUse=testIdx{c}; end
                mu_t = (mean(Xc{c}(tw,:,trUse),[1 3],"omitnan") - muZ)./sdZ;
                traj2D_r(r,c,ti,1) = mu_t*wP;
                traj2D_r(r,c,ti,2) = mu_t*wR;
            end

            % ---- Mean projection per condition at this time bin ----
            for c = 1:4
                rawP_test(c,ti)  = mean(F{c}     * wP, "omitnan");
                rawP_train(c,ti) = mean(Ftrain{c} * wP, "omitnan");
                rawR_test(c,ti)  = mean(F{c}     * wR, "omitnan");
                rawR_train(c,ti) = mean(Ftrain{c} * wR, "omitnan");
            end

            % ---- Percent class-1: time-resolved decision boundary ----
            % The threshold at each time bin is the midpoint between the
            % TRAIN-SET class-mean projections AT THAT SAME TIME BIN.
            %
            % Why time-resolved rather than fixed:
            %   A fixed threshold computed from the training window fails at
            %   other time points because the absolute level of neural activity
            %   shifts across the trial (stimulus-evoked drive, adaptation,
            %   etc.). After baseline subtraction and z-scoring relative to
            %   the training window, pre-stimulus activity sits at a
            %   systematically different absolute value than training-window
            %   activity, so any fixed threshold produces a bias.
            %
            %   Using the train-set class means AT the current time bin as the
            %   boundary removes this offset: when all conditions look the same
            %   (e.g. pre-stimulus), all class means coincide and every
            %   condition lands at ~50%. When the classes separate (during /
            %   after the training window), the boundary tracks the separation
            %   and the measure reflects true discriminability on test trials.
            %
            % Pitch axis: class-0 = conds {1,2} (pitch band 0)
            %             class-1 = conds {3,4} (pitch band 1)
            thresh_P_ti = (mean([rawP_train(1,ti), rawP_train(2,ti)]) + ...
                           mean([rawP_train(3,ti), rawP_train(4,ti)])) / 2;

            % Rule axis:  class-0 = conds {1,3} (ABCD)
            %             class-1 = conds {2,4} (ABBA)
            thresh_R_ti = (mean([rawR_train(1,ti), rawR_train(3,ti)]) + ...
                           mean([rawR_train(2,ti), rawR_train(4,ti)])) / 2;

            for c = 1:4
                p1P_test_r(r,c,ti)  = mean(F{c}     * wP > thresh_P_ti, "omitnan");
                p1P_train_r(r,c,ti) = mean(Ftrain{c} * wP > thresh_P_ti, "omitnan");
                p1R_test_r(r,c,ti)  = mean(F{c}     * wR > thresh_R_ti, "omitnan");
                p1R_train_r(r,c,ti) = mean(Ftrain{c} * wR > thresh_R_ti, "omitnan");
            end
        end  % ti

        % ---- Normalisation: scale anchored to TRAINING-set projections ----
        % The normalisation scale (centre and half-range) is derived from
        % the TRAIN projections at the LDA training window.  Using the
        % training set is critical: the LDA axis was optimised to maximise
        % separation on those trials, so the class means there are always
        % well-separated and the half-range is never near zero.  Using the
        % (small) test set instead causes the half-range to be noisy and
        % sometimes near zero, blowing up the normalised values.
        %
        % The same fixed linear scale is then applied to BOTH test and
        % train projections at every time bin, so the time courses share a
        % common axis: at the training window the train-set class means sit
        % at exactly -1 and +1 by construction.
        %
        %   Pitch axis: class-0 = conds {1,2}  class-1 = conds {3,4}
        %   Rule  axis: class-0 = conds {1,3}  class-1 = conds {2,4}

        scP = norm_scale_from_ref(rawP_train, [1 2],[3 4], idxPitch);
        scR = norm_scale_from_ref(rawR_train, [1 3],[2 4], idxRule);

        pnP_test_r(r,:,:)  = apply_norm_scale(rawP_test,  scP);
        pnP_train_r(r,:,:) = apply_norm_scale(rawP_train, scP);
        pnR_test_r(r,:,:)  = apply_norm_scale(rawR_test,  scR);
        pnR_train_r(r,:,:) = apply_norm_scale(rawR_train, scR);

        % ---- Rectangle centroids (kept from v3) ----
        for c = 1:4
            muXt = (mean(Xc{c}(idxRectPitch,:,testIdx{c}),[1 3],"omitnan")-muZ)./sdZ;
            muYt = (mean(Xc{c}(idxRectRule, :,testIdx{c}),[1 3],"omitnan")-muZ)./sdZ;
            pts_r_test(r,c,1)  = muXt*wP;  pts_r_test(r,c,2)  = muYt*wR;

            muXf = (mean(Xc{c}(idxRectPitch,:,fitIdx{c}),[1 3],"omitnan")-muZ)./sdZ;
            muYf = (mean(Xc{c}(idxRectRule, :,fitIdx{c}),[1 3],"omitnan")-muZ)./sdZ;
            pts_r_train(r,c,1) = muXf*wP;  pts_r_train(r,c,2) = muYf*wR;
        end

    end  % CV splits

    % ---- average over CV splits ----
    aucPP(s,:) = mean(aucPP_r,1,"omitnan");
    aucRR(s,:) = mean(aucRR_r,1,"omitnan");
    aucPR(s,:) = mean(aucPR_r,1,"omitnan");
    aucRP(s,:) = mean(aucRP_r,1,"omitnan");

    ptsTest  = squeeze(mean(pts_r_test, 1,"omitnan"));
    ptsTrain = squeeze(mean(pts_r_train,1,"omitnan"));
    pts_rect_test(s,:,:)  = ptsTest;
    pts_rect_train(s,:,:) = ptsTrain;
    traj2D(s,:,:,:) = squeeze(mean(traj2D_r,1,"omitnan"));

    % geometry scores
    p0=ptsTest(1,:); p1=ptsTest(2,:); p2=ptsTest(3,:); p3=ptsTest(4,:);
    cP=cos_sim(p2-p0,p3-p1); cR=cos_sim(p1-p0,p3-p2);
    pitch_par_cos(s)=cP; rule_par_cos(s)=cR;
    pitch_par_abs(s)=abs(cP); rule_par_abs(s)=abs(cR);
    if opt.allowAntiparallel, rect_score(s)=mean([abs(cP),abs(cR)],"omitnan");
    else, rect_score(s)=mean([cP,cR],"omitnan"); end

    % Store averaged projections [nS x 4 x nT]
    proj_norm_pitch_test(s,:,:)  = squeeze(mean(pnP_test_r, 1,"omitnan"));
    proj_norm_pitch_train(s,:,:) = squeeze(mean(pnP_train_r,1,"omitnan"));
    proj_norm_rule_test(s,:,:)   = squeeze(mean(pnR_test_r, 1,"omitnan"));
    proj_norm_rule_train(s,:,:)  = squeeze(mean(pnR_train_r,1,"omitnan"));

    pct1_pitch_test(s,:,:)  = squeeze(mean(p1P_test_r, 1,"omitnan"));
    pct1_pitch_train(s,:,:) = squeeze(mean(p1P_train_r,1,"omitnan"));
    pct1_rule_test(s,:,:)   = squeeze(mean(p1R_test_r, 1,"omitnan"));
    pct1_rule_train(s,:,:)  = squeeze(mean(p1R_train_r,1,"omitnan"));

    valid(s) = true;
end  % sessions

keep = find(valid);
if isempty(keep), error("No valid sessions."); end

% ================================================================
%  PACKAGE OUTPUTS
% ================================================================
Out = struct();
Out.options       = opt;
Out.t             = t;
Out.nSessions     = numel(keep);
Out.sessionIDs    = dateSessions(valid);
Out.sessionAnimal = sessAnimal(valid);
Out.nUnits        = nUnitsSess(valid);
Out.conditionNames = patterns;  % ["ABCD0","ABBA0","ABCD1","ABBA1"]

Out.points_rect_test_session  = pts_rect_test(valid,:,:);
Out.points_rect_train_session = pts_rect_train(valid,:,:);
Out.points_rect_session       = Out.points_rect_test_session;  % back-compat

Out.aucPitch_onPitch_mean = mean(aucPP(valid,:),1,"omitnan");
Out.aucPitch_onPitch_sem  = std( aucPP(valid,:),0,1,"omitnan") / sqrt(numel(keep));
Out.aucRule_onRule_mean   = mean(aucRR(valid,:),1,"omitnan");
Out.aucRule_onRule_sem    = std( aucRR(valid,:),0,1,"omitnan") / sqrt(numel(keep));

Out.traj2D_mean = squeeze(mean(traj2D(valid,:,:,:),1,"omitnan"));

% Time-resolved projections  [nSess x 4 x nT]
Out.proj_norm_pitch_test  = proj_norm_pitch_test(valid,:,:);
Out.proj_norm_pitch_train = proj_norm_pitch_train(valid,:,:);
Out.proj_norm_rule_test   = proj_norm_rule_test(valid,:,:);
Out.proj_norm_rule_train  = proj_norm_rule_train(valid,:,:);
Out.pct1_pitch_test       = pct1_pitch_test(valid,:,:);   % [0,1]
Out.pct1_pitch_train      = pct1_pitch_train(valid,:,:);
Out.pct1_rule_test        = pct1_rule_test(valid,:,:);
Out.pct1_rule_train       = pct1_rule_train(valid,:,:);

% ================================================================
%  PLOTTING
% ================================================================
if opt.plot
    if opt.forceFigureVisible, set(groot,'DefaultFigureVisible','on'); end

    % 1. AUC
    if opt.plotAUC
        figure('Name','AUC','NumberTitle','off'); hold on
        shaded_sem(t, Out.aucPitch_onPitch_mean, Out.aucPitch_onPitch_sem, [0 0.447 0.741]);
        shaded_sem(t, Out.aucRule_onRule_mean,   Out.aucRule_onRule_sem,   [0.85 0.325 0.098]);
        yline(0.5,'k--'); xlabel("Time (s)"); ylabel("AUC");
        title("Decoding (Test)");
        legend(["Pitch on Pitch","Rule on Rule"],'Location','best');
        grid on; hold off;
    end

    % 2. Rectangle / geometry
    if opt.plotRectangle
        ptsAll = Out.points_rect_test_session;
        figure('Name','Geometry','NumberTitle','off'); hold on
        colors = lines(4);
        for c = 1:4
            scatter(ptsAll(:,c,1), ptsAll(:,c,2), 30, colors(c,:), 'filled', ...
                'MarkerFaceAlpha',0.3,'MarkerEdgeColor','none','HandleVisibility','off');
        end
        if opt.plotEllipses
            for c=1:4, plot_cov_ellipse(ptsAll(:,c,1),ptsAll(:,c,2),1.0,colors(c,:)); end
        end
        mu   = squeeze(mean(ptsAll,1,"omitnan"));
        ords = [1 2;3 4;1 3;2 4];
        for k=1:4
            plot([mu(ords(k,1),1) mu(ords(k,2),1)],[mu(ords(k,1),2) mu(ords(k,2),2)],...
                'k-','LineWidth',1.5);
        end
        for c=1:4
            scatter(mu(c,1),mu(c,2),100,colors(c,:),'filled',...
                'MarkerEdgeColor','k','DisplayName',char(patterns(c)));
        end
        xlabel("Pitch Axis"); ylabel("Rule Axis");
        title(sprintf("Geometry (Test) | Score=%.2f", mean(rect_score(valid))));
        legend('Location','best'); grid on; axis tight; hold off;
    end

    % 3. Time-resolved projection
    if opt.plotProjection
        plot_projection_timeseries(Out, opt, patterns);
    end
end
end  % main function


% ================================================================
%  PLOTTING HELPER: time-resolved projection
% ================================================================
function plot_projection_timeseries(Out, opt, patterns)

t    = Out.t;
mode = opt.projectionMode;

% Condition colours: ABCD0, ABBA0, ABCD1, ABBA1
cmap = [  0   0   0; ... % ABCD0 - black
         20 125 125; ... % ABBA0 - teal
        232 115 155; ... % ABCD1 - pink
        115 232 193] / 255;  % ABBA1 - mint

pipStarts = [0 0.25 0.50 0.75];

datasets = {};
if any(opt.projectionWhich == ["test","both"]),  datasets{end+1} = "test";  end
if any(opt.projectionWhich == ["train","both"]), datasets{end+1} = "train"; end

for di = 1:numel(datasets)
    dsLabel = datasets{di};

    if mode == "normalized"
        pitchData = Out.(sprintf("proj_norm_pitch_%s", dsLabel));  % [nSess x 4 x nT]
        ruleData  = Out.(sprintf("proj_norm_rule_%s",  dsLabel));
        yLabelP   = "Normalised projection (pitch axis, a.u.)";
        yLabelR   = "Normalised projection (rule axis, a.u.)";
        refLine   = 0;
        yLimP     = [];   % auto
        yLimR     = [];
    else  % percent_class1
        pitchData = Out.(sprintf("pct1_pitch_%s", dsLabel)) * 100;  % -> %
        ruleData  = Out.(sprintf("pct1_rule_%s",  dsLabel)) * 100;
        yLabelP   = "% trials: pitch band 1 (pitch axis)";
        yLabelR   = "% trials: ABBA (rule axis)";
        refLine   = 50;
        yLimP     = [0 100];
        yLimR     = [0 100];
    end

    figTitle = sprintf("Time-resolved projection  [%s | %s]", dsLabel, strrep(mode,'_',' '));
    hf = figure('Name', figTitle, 'NumberTitle', 'off');
    set(hf, 'Position', [80 150 1100 460]);

    ax1 = subplot(1,2,1);
    draw_proj_ts(ax1, t, pitchData, patterns, cmap, ...
        "Projection onto Pitch LDA axis", yLabelP, yLimP, refLine, ...
        opt.pitchTrainWindow, pipStarts, ...
        opt.projectionShowSessions, opt.projectionShowPips, ...
        "Pitch band 0 (negative)   |   Pitch band 1 (positive)");

    ax2 = subplot(1,2,2);
    draw_proj_ts(ax2, t, ruleData, patterns, cmap, ...
        "Projection onto Rule LDA axis", yLabelR, yLimR, refLine, ...
        opt.ruleTrainWindow, pipStarts, ...
        opt.projectionShowSessions, opt.projectionShowPips, ...
        "ABCD (negative)   |   ABBA (positive)");

    sgtitle(figTitle, 'FontSize', 13, 'FontWeight', 'bold');
end
end


function draw_proj_ts(ax, t, data, patterns, cmap, titleStr, yLabelStr, yLim, refLine, trainWin, pipStarts, showSess, showPips, signNote)
%DRAW_PROJ_TS  One projection panel: mean +/- SEM time series, one line per condition.
%   data - [nSess x 4 x nT]

nSess = size(data,1);

hold(ax,'on');

% auto y-limits for background shading
allVals = data(:);
allVals = allVals(isfinite(allVals));
if ~isempty(yLim)
    yl = yLim;
else
    pad = 0.12 * (max(allVals)-min(allVals));
    yl  = [min(allVals)-pad, max(allVals)+pad];
    if yl(1)==yl(2), yl = yl + [-0.5 0.5]; end
end

% grey shading for the axis training window
patch(ax, [trainWin(1) trainWin(2) trainWin(2) trainWin(1)], ...
    [yl(1) yl(1) yl(2) yl(2)], [0.82 0.82 0.82], ...
    'EdgeColor','none','FaceAlpha',0.40,'HandleVisibility','off');

% pip onset markers
if showPips
    for p = pipStarts
        xline(ax, p, '--', 'Color',[0.55 0.55 0.55], ...
            'LineWidth',0.8,'HandleVisibility','off');
    end
end

% stimulus onset
xline(ax, 0, 'k-', 'LineWidth',0.8,'HandleVisibility','off');

% reference (0 or 50%)
yline(ax, refLine, 'k:', 'LineWidth',1.2,'HandleVisibility','off');

% per-condition traces
for c = 1:4
    d_c = squeeze(data(:,c,:));   % [nSess x nT]
    if nSess == 1, d_c = d_c(:).'; end  % keep as row vector

    mu  = mean(d_c, 1, "omitnan");
    sem = std( d_c, 0, 1, "omitnan") / sqrt(nSess);

    if showSess
        for si = 1:nSess
            plot(ax, t, d_c(si,:), 'Color', [cmap(c,:) 0.18], ...
                'LineWidth',0.6,'HandleVisibility','off');
        end
    end

    % shaded SEM band
    patch(ax, [t(:); flipud(t(:))], [mu(:)-sem(:); flipud(mu(:)+sem(:))], ...
        cmap(c,:), 'FaceAlpha',0.20,'EdgeColor','none','HandleVisibility','off');

    % mean line
    plot(ax, t, mu, 'Color', cmap(c,:), 'LineWidth', 2.2, ...
        'DisplayName', char(patterns(c)));
end

% cosmetics
set(ax,'Box','off','TickDir','out','FontSize',10);
xlabel(ax,'Time (s)');
ylabel(ax, yLabelStr);
title(ax, titleStr, 'FontWeight','bold');
ylim(ax, yl);
xlim(ax, [t(1) t(end)]);
legend(ax,'Location','best','FontSize',9);

% sign note bottom-right
text(ax, 0.98, 0.03, signNote, 'Units','normalized', ...
    'HorizontalAlignment','right','VerticalAlignment','bottom', ...
    'FontSize',8,'Color',[0.4 0.4 0.4],'Interpreter','none');

hold(ax,'off');
end


% ================================================================
%  NORMALISATION HELPERS
% ================================================================
function sc = norm_scale_from_ref(rawMat, idx0, idx1, refBins)
%NORM_SCALE_FROM_REF  Compute linear scaling so that, averaged over
%   refBins, class-0 conditions map to -1 and class-1 to +1.
%   rawMat - [4 x nT]
m0 = mean(rawMat(idx0, refBins), "all", "omitnan");
m1 = mean(rawMat(idx1, refBins), "all", "omitnan");
sc.centre     = (m0 + m1) / 2;
sc.half_range = (m1 - m0) / 2;
if abs(sc.half_range) < eps, sc.half_range = 1; end  % guard div/0
end

function normed = apply_norm_scale(rawMat, sc)
%APPLY_NORM_SCALE  Apply pre-computed scale to [4 x nT] matrix.
normed = (rawMat - sc.centre) / sc.half_range;
end


% ================================================================
%  SHARED HELPERS (unchanged from v3)
% ================================================================
function plot_cov_ellipse(x, y, nsd, col)
x=x(:); y=y(:); good=isfinite(x)&isfinite(y); x=x(good); y=y(good);
if numel(x)<5, return; end
mu=[mean(x),mean(y)]; C=cov([x y],1);
[V,D]=eig(C); [d,ord]=sort(diag(D),'descend'); V=V(:,ord);
theta=linspace(0,2*pi,100);
ell=(V*[nsd*sqrt(d(1))*cos(theta); nsd*sqrt(d(2))*sin(theta)])';
plot(mu(1)+ell(:,1),mu(2)+ell(:,2),'-','Color',[col 0.5],...
    'LineWidth',1,'HandleVisibility','off');
end

function Xfeat = feat_trials(X, idxT, trIdx)
mu = squeeze(mean(X(idxT,:,trIdx),1,"omitnan")); Xfeat = mu.';
end

function [Xb,yb] = balance_binary(X,y)
y=y(:); i0=find(y==0); i1=find(y==1);
n=min(numel(i0),numel(i1));
i0=i0(randperm(numel(i0),n)); i1=i1(randperm(numel(i1),n));
idx=[i0;i1]; idx=idx(randperm(numel(idx))); Xb=X(idx,:); yb=y(idx);
end

function w = lda_axis(X,y,mode,lambda,sdFloor)
X=double(X); y=y(:); X0=X(y==0,:); X1=X(y==1,:);
dmu=(mean(X1,1)-mean(X0,1)).';
Sw=((size(X0,1))*cov(X0,1)+(size(X1,1))*cov(X1,1))/(size(X0,1)+size(X1,1));
w=(Sw+lambda*trace(Sw)/size(Sw,1)*eye(size(Sw,1)))\dmu;
end

function idxCell = make_time_windows(t,winSec,mode)
dt=mean(diff(t)); bins=max(1,round(winSec/dt));
for ti=1:numel(t)
    if mode=="causal", idxCell{ti}=max(1,ti-bins+1):ti;
    else, idxCell{ti}=max(1,ti-round(bins/2)):min(numel(t),ti+round(bins/2)); end
end
end

function auc=auc_rank(y01,score)
[~,p]=sort(score); r=zeros(size(score)); r(p)=1:numel(score);
n1=sum(y01); n0=sum(~y01);
auc=(sum(r(y01==1))-n1*(n1+1)/2)/(n1*n0);
end

function shaded_sem(x,m,s,col)
x=x(:); m=m(:); s=s(:);
patch([x;flipud(x)],[m-s;flipud(m+s)],col,'FaceAlpha',0.2,'EdgeColor','none');
plot(x,m,'Color',col,'LineWidth',2);
end

function c=cos_sim(a,b)
c=(a(:).'*b(:))/(norm(a)*norm(b));
end
