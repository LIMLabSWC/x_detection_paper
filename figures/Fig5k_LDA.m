clc
clear all

pth = '/Users/athena/Work/Projects/Dammy/ephys analysis/';
h5file = [pth,'/unit_nozscore_file.h5'];
idx = h5ephys_loader(h5file);

idx.animals          % list all animals
idx.summary          % how many A-* datasets per animal


Out = lda_axes_statespace(idx, "all", ...
  "pitchTrainWindow",[0.00 0.50], ...
  "ruleTrainWindow",[0.55 1.4], ...
  "integrateWinRuleTime",0.15, ...
  "integrateWinProj",0.10, ...
  "integrateMode","causal", ...
  "plot",true);

save("lda_pitch_rule_leakage_Pitch0to500_Rule750to1_Out.mat", "Out")

Dat.aucPitch_onPitch_mean = Out.aucPitch_onPitch_mean;
Dat.aucPitch_onPitch_sem = Out.aucPitch_onPitch_sem;

Dat.aucPitch_onRule_mean = Out.aucPitch_onRule_mean;
Dat.aucPitch_onRule_sem = Out.aucPitch_onRule_sem;

Dat.aucRule_onPitch_mean = Out.aucRule_onPitch_mean;
Dat.aucRule_onPitch_sem = Out.aucRule_onPitch_sem;

Dat.aucRule_onRule_mean = Out.aucRule_onRule_mean;
Dat.aucRule_onRule_sem = Out.aucRule_onRule_sem;

save("lda_pitch_rule_leakage_Pitch0to500_Rule750to1_Dat.mat", "Dat")

function Out = lda_axes_statespace(idx, animalID, varargin)
%LDA_AXES_STATESPACE  LDA pitch/rule axes + angle + state-space + 1D leakage AUC timecourses.
%
% Adds:
%   Out.aucPitch_onPitchAxis(t)  : AUC(low vs high) using score = x·wPitch
%   Out.aucPitch_onRuleAxis(t)   : leakage
%   Out.aucRule_onRuleAxis(t)    : AUC(ABCD vs ABBA) using score = x·wRuleLate
%   Out.aucRule_onPitchAxis(t)   : leakage
%
% Interpretation:
%   - Factorized coding: Pitch separable along pitch axis, Rule separable along rule axis,
%     while cross-axis AUC stays near 0.5 (little leakage).

% ---- parse options (same as v2) ----
p = inputParser;

p.addParameter("dt", 0.01);
p.addParameter("tStart", -0.5);
p.addParameter("tEnd", 1.5);

p.addParameter("baselineWindow", [-0.50 0.00]);
p.addParameter("pitchTrainWindow", [0.00 0.50]);
p.addParameter("ruleTrainWindow",  [0.55 1.05]);

p.addParameter("doRuleAxisTimeResolved", true);
p.addParameter("integrateWinRuleTime", 0.15);
p.addParameter("integrateMode", "causal");

p.addParameter("integrateWinProj", 0.05);

p.addParameter("doBaseline", true);
p.addParameter("zscoreNeurons", true);
p.addParameter("sdFloor", 1e-3);
p.addParameter("minUnits", 10);

p.addParameter("ldaMode", "ridge");
p.addParameter("lambda", 0.10);
p.addParameter("balanceTrain", true);

p.addParameter("nDiscardSessions", 0);

p.addParameter("discardFirstGlobalTrials", 0);
p.addParameter("trialGlobalSuffix", "_trialGlobal");

p.addParameter("plot", true);
p.addParameter("showPips", true);
p.addParameter("plotTrajectories", true);
p.addParameter("plotRectangleScatter", true);
p.addParameter("plotLeakageAUC", true);

p.parse(varargin{:});
opt = p.Results;

if iscell(animalID), animalID = string(animalID); end
animalID = string(animalID);
opt.integrateMode = lower(string(opt.integrateMode));
opt.ldaMode = lower(string(opt.ldaMode));

% ---- time ----
t = opt.tStart:opt.dt:opt.tEnd;
assert(numel(t)==201, "Expected 201 bins for -0.5:0.01:1.5.");

idxB     = find(t >= opt.baselineWindow(1) & t < opt.baselineWindow(2));
idxPitch = find(t >= opt.pitchTrainWindow(1) & t < opt.pitchTrainWindow(2));
idxRule  = find(t >= opt.ruleTrainWindow(1)  & t < opt.ruleTrainWindow(2));

if opt.doBaseline
    assert(~isempty(idxB), "baselineWindow has no bins.");
end
assert(~isempty(idxPitch) && ~isempty(idxRule), "Training windows have no bins.");

pipStarts = [0 0.25 0.50 0.75];
pipDur = 0.15;

ruleTimeIdx = make_time_windows(t, opt.integrateWinRuleTime, opt.integrateMode);
projTimeIdx = make_time_windows(t, opt.integrateWinProj, opt.integrateMode);

% ---- conditions ----
cond = table( ...
    ["A-0"; "A-1"; "A-2"; "A-3"], ...
    ["ABCD0";"ABBA0";"ABCD1";"ABBA1"], ...
    [0;0;1;1], ...
    [+1;-1;+1;-1], ...
    'VariableNames', {'AFile','Pattern','Band','Rule'});
patterns = string(cond.Pattern);

% ---- select sessions ----
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
    animals = unique(T.Animal);
    for a = animals'
        dsList = unique(T.DateSession(T.Animal==a));
        dsList = sort(dsList);
        nDrop = min(opt.nDiscardSessions, numel(dsList));
        keepSessions = [keepSessions; dsList(nDrop+1:end)]; %#ok<AGROW>
    end
    T = T(ismember(T.DateSession, keepSessions), :);
end
dateSessions = unique(T.DateSession);

% ---- per-session storage ----
valid = false(numel(dateSessions),1);
sessAnimal = strings(numel(dateSessions),1);
nUnitsSess = nan(numel(dateSessions),1);

wPitch_sess = cell(numel(dateSessions),1);
wRuleLate_sess = cell(numel(dateSessions),1);
wRuleTime_sess = cell(numel(dateSessions),1);
traj_sess = cell(numel(dateSessions),1);

% NEW: 1D AUC timecourses per session
aucPitch_onPitch = nan(numel(dateSessions), numel(t));
aucPitch_onRule  = nan(numel(dateSessions), numel(t));
aucRule_onRule   = nan(numel(dateSessions), numel(t));
aucRule_onPitch  = nan(numel(dateSessions), numel(t));

for s = 1:numel(dateSessions)
    ds = dateSessions(s);
    Ts = T(T.DateSession==ds,:);
    sessAnimal(s) = extractBefore(string(ds), "_");

    % load 4 conditions
    Xc = cell(4,1);
    ok = true;
    for c = 1:4
        rr = Ts(Ts.AFile==cond.AFile(c), :);
        if isempty(rr), ok=false; break; end
        APath = string(rr.APath(1));
        X = double(idx.read(APath)); % time x units x trials
        if size(X,1) ~= numel(t), error("Time mismatch in %s", APath); end

        if opt.discardFirstGlobalTrials > 0
            if ~isfield(idx,"h5file")
                error("discardFirstGlobalTrials>0 requires idx.h5file.");
            end
            metaPath = APath + string(opt.trialGlobalSuffix);
            g = double(h5read(idx.h5file, metaPath));
            keep = g(:) > opt.discardFirstGlobalTrials;
            if sum(keep) < 4, ok=false; break; end
            X = X(:,:,keep);
        end

        if opt.doBaseline
            base = mean(X(idxB,:,:),1);
            X = X - base;
        end

        Xc{c} = X;
    end
    if ~ok, continue; end

    nUnits = size(Xc{1},2);
    if any(cellfun(@(Z) size(Z,2)~=nUnits, Xc))
        error("Unit count differs across conditions in session %s", string(ds));
    end
    if nUnits < opt.minUnits, continue; end
    nUnitsSess(s) = nUnits;

    % ===== training features =====
    Xlow  = [features_window(Xc{1}, idxPitch); features_window(Xc{2}, idxPitch)];
    Xhigh = [features_window(Xc{3}, idxPitch); features_window(Xc{4}, idxPitch)];
    yPitch = [zeros(size(Xlow,1),1); ones(size(Xhigh,1),1)];
    XPitch = [Xlow; Xhigh];

    Xabcdn = [features_window(Xc{1}, idxRule); features_window(Xc{3}, idxRule)];
    Xabban = [features_window(Xc{2}, idxRule); features_window(Xc{4}, idxRule)];
    yRuleLate = [zeros(size(Xabcdn,1),1); ones(size(Xabban,1),1)]; % ABBA=1
    XRuleLate = [Xabcdn; Xabban];

    % zscore within session using pooled training
    if opt.zscoreNeurons
        Xpool = [XPitch; XRuleLate];
        muZ = mean(Xpool,1);
        sdZ = std(Xpool,0,1);
        sdZ = max(sdZ, opt.sdFloor);

        XPitch = (XPitch - muZ) ./ sdZ;
        XRuleLate = (XRuleLate - muZ) ./ sdZ;
    else
        muZ = zeros(1,nUnits);
        sdZ = ones(1,nUnits);
    end

    % balance for axis estimation
    if opt.balanceTrain
        [XPitchB, yPitchB] = balance_binary(XPitch, yPitch);
        [XRuleB, yRuleB]   = balance_binary(XRuleLate, yRuleLate);
    else
        XPitchB = XPitch; yPitchB = yPitch;
        XRuleB  = XRuleLate; yRuleB = yRuleLate;
    end

    wPitch = lda_axis(XPitchB, yPitchB, opt.ldaMode, opt.lambda, opt.sdFloor);
    wRuleLate = lda_axis(XRuleB, yRuleB, opt.ldaMode, opt.lambda, opt.sdFloor);

    wPitch = wPitch / max(norm(wPitch), eps);
    wRuleLate = wRuleLate / max(norm(wRuleLate), eps);

    % sign align
    projPitchTrain = XPitch * wPitch;
    if mean(projPitchTrain(yPitch==1)) < mean(projPitchTrain(yPitch==0)), wPitch = -wPitch; end
    projRuleTrain  = XRuleLate * wRuleLate;
    if mean(projRuleTrain(yRuleLate==1)) < mean(projRuleTrain(yRuleLate==0)), wRuleLate = -wRuleLate; end

    % time-resolved rule axis (optional)
    if opt.doRuleAxisTimeResolved
        wRuleTime = nan(nUnits, numel(t));
        for ti = 1:numel(t)
            tw = ruleTimeIdx{ti};
            Xabcdn_t = [features_window(Xc{1}, tw); features_window(Xc{3}, tw)];
            Xabban_t = [features_window(Xc{2}, tw); features_window(Xc{4}, tw)];
            Xt = [Xabcdn_t; Xabban_t];
            yt = [zeros(size(Xabcdn_t,1),1); ones(size(Xabban_t,1),1)];

            Xt = (Xt - muZ) ./ sdZ;
            if opt.balanceTrain
                [XtB, ytB] = balance_binary(Xt, yt);
            else
                XtB = Xt; ytB = yt;
            end

            wt = lda_axis(XtB, ytB, opt.ldaMode, opt.lambda, opt.sdFloor);
            wt = wt / max(norm(wt), eps);

            % sign align to ABBA=+ at that time (not using late axis)
            sc = Xt * wt;
            if mean(sc(yt==1)) < mean(sc(yt==0)), wt = -wt; end

            wRuleTime(:,ti) = wt;
        end
    else
        wRuleTime = [];
    end

    % trajectories (condition mean)
    traj = struct();
    for c = 1:4
        X = Xc{c};
        XY = nan(numel(t),2);
        for ti = 1:numel(t)
            tw = projTimeIdx{ti};
            mu_t = mean(X(tw,:,:), [1 3], "omitnan");
            mu_t = (mu_t - muZ) ./ sdZ;
            mu_t = mu_t(:);
            XY(ti,1) = mu_t.' * wPitch;
            XY(ti,2) = mu_t.' * wRuleLate;
        end
        traj.(patterns(c)) = XY;
    end

    % ===== NEW: leakage AUC timecourses =====
    % build pooled trial matrices per time for pitch and rule labels, then score on each axis.
    for ti = 1:numel(t)
        tw = projTimeIdx{ti};

        % features at time for each condition
        F0 = features_window(Xc{1}, tw); % ABCD0
        F1 = features_window(Xc{2}, tw); % ABBA0
        F2 = features_window(Xc{3}, tw); % ABCD1
        F3 = features_window(Xc{4}, tw); % ABBA1

        % zscore with same muZ/sdZ
        F0 = (F0 - muZ) ./ sdZ; F1 = (F1 - muZ) ./ sdZ;
        F2 = (F2 - muZ) ./ sdZ; F3 = (F3 - muZ) ./ sdZ;

        % pitch task labels: low(0)=F0+F1, high(1)=F2+F3
        Xp = [F0;F1;F2;F3];
        yp = [zeros(size(F0,1)+size(F1,1),1); ones(size(F2,1)+size(F3,1),1)];

        % rule task labels: ABCD(0)=F0+F2, ABBA(1)=F1+F3
        Xr = [F0;F2;F1;F3];
        yr = [zeros(size(F0,1)+size(F2,1),1); ones(size(F1,1)+size(F3,1),1)];

        % scores on each axis
        sp = Xp * wPitch;  % pitch axis score for pitch task
        sr_onP = Xr * wPitch; % pitch axis score for rule task (leakage)
        sr = Xr * wRuleLate;  % rule axis score for rule task
        sp_onR = Xp * wRuleLate; % rule axis score for pitch task (leakage)

        aucPitch_onPitch(s,ti) = auc_rank(yp, sp);
        aucPitch_onRule(s,ti)  = auc_rank(yp, sp_onR);
        aucRule_onRule(s,ti)   = auc_rank(yr, sr);
        aucRule_onPitch(s,ti)  = auc_rank(yr, sr_onP);
    end

    % save
    wPitch_sess{s} = wPitch;
    wRuleLate_sess{s} = wRuleLate;
    wRuleTime_sess{s} = wRuleTime;
    traj_sess{s} = traj;
    valid(s) = true;
end

% keep valid
keep = find(valid);
if isempty(keep), error("No valid sessions after filtering."); end

Out = struct();
Out.options = opt;
Out.t = t;
Out.sessionAnimal = sessAnimal(valid);
Out.sessionIDs = dateSessions(valid);
Out.nSessions = numel(keep);
Out.nUnits = nUnitsSess(valid);

Out.pitchAxis_perSession = wPitch_sess(valid);
Out.ruleAxisLate_perSession = wRuleLate_sess(valid);
Out.ruleAxisTime_perSession = wRuleTime_sess(valid);

% angles
angleLate = nan(Out.nSessions,1);
angleTime = nan(Out.nSessions, numel(t));

for i = 1:Out.nSessions
    wP = Out.pitchAxis_perSession{i};
    wRL = Out.ruleAxisLate_perSession{i};
    angleLate(i) = angle_between(wP, wRL, true);
    if opt.doRuleAxisTimeResolved
        WRt = Out.ruleAxisTime_perSession{i};
        for ti = 1:numel(t)
            angleTime(i,ti) = angle_between(wP, WRt(:,ti), true);
        end
    end
end

Out.angleLate = angleLate;
Out.angleLate_mean = mean(angleLate,"omitnan");
Out.angleLate_sem  = std(angleLate,0,"omitnan") / sqrt(Out.nSessions);

if opt.doRuleAxisTimeResolved
    Out.angleTime_session = angleTime;
    Out.angleTime_mean = mean(angleTime,1,"omitnan");
    Out.angleTime_sem  = std(angleTime,0,1,"omitnan") ./ sqrt(max(1,sum(isfinite(angleTime),1)));
end

% trajectories mean/sem
patterns = ["ABCD0","ABBA0","ABCD1","ABBA1"];
trajMean = struct(); trajSem = struct();
for c = 1:4
    P = patterns(c);
    XYall = nan(Out.nSessions, numel(t), 2);
    for i = 1:Out.nSessions
        tr = traj_sess{keep(i)};
        XYall(i,:,:) = tr.(P);
    end
    trajMean.(P) = squeeze(mean(XYall,1,"omitnan"));
    trajSem.(P)  = squeeze(std(XYall,0,1,"omitnan")) ./ sqrt(Out.nSessions);
end
Out.trajMean = trajMean;
Out.trajSem = trajSem;

% leakage AUC mean/sem
Out.aucPitch_onPitch_session = aucPitch_onPitch(valid,:);
Out.aucPitch_onRule_session  = aucPitch_onRule(valid,:);
Out.aucRule_onRule_session   = aucRule_onRule(valid,:);
Out.aucRule_onPitch_session  = aucRule_onPitch(valid,:);

[Out.aucPitch_onPitch_mean, Out.aucPitch_onPitch_sem] = mean_sem(Out.aucPitch_onPitch_session);
[Out.aucPitch_onRule_mean,  Out.aucPitch_onRule_sem]  = mean_sem(Out.aucPitch_onRule_session);
[Out.aucRule_onRule_mean,   Out.aucRule_onRule_sem]   = mean_sem(Out.aucRule_onRule_session);
[Out.aucRule_onPitch_mean,  Out.aucRule_onPitch_sem]  = mean_sem(Out.aucRule_onPitch_session);

% ---- plots ----
if opt.plot
    % angle
    if opt.doRuleAxisTimeResolved
        figure; hold on
        if opt.showPips, draw_pips(pipStarts, pipDur); end
        shaded_sem(t, Out.angleTime_mean, Out.angleTime_sem);
        yline(90,"k--","HandleVisibility","off");
        xlabel("Time (s)"); ylabel("Angle (deg)");
        title(sprintf("Pitch vs Rule axis angle | nSessions=%d", Out.nSessions));
        grid on; ylim([0 180]); hold off
    end

    % trajectories
    if opt.plotTrajectories
        figure; hold on
        for c = 1:4
            P = patterns(c);
            XY = Out.trajMean.(P);
            plot(XY(:,1), XY(:,2), "LineWidth",2, "DisplayName", char(P));
        end
        xlabel("Pitch axis"); ylabel("Rule axis");
        title(sprintf("State-space trajectories (session-wise axes) | nSessions=%d", Out.nSessions));
        legend("Location","best"); grid on; axis tight; hold off
    end

    % rectangle scatter
    if opt.plotRectangleScatter
        figure; hold on
        idxLate = idxRule;
        pts = nan(4,2);
        for c = 1:4
            P = patterns(c);
            XY = Out.trajMean.(P);
            pts(c,:) = mean(XY(idxLate,:), 1, "omitnan");
            scatter(pts(c,1), pts(c,2), 70, "filled", "DisplayName", char(P));
        end
        iABCD0 = 1; iABBA0 = 2; iABCD1 = 3; iABBA1 = 4;
        plot([pts(iABCD0,1) pts(iABCD1,1)], [pts(iABCD0,2) pts(iABCD1,2)], "k-", "LineWidth",1.5, "HandleVisibility","off");
        plot([pts(iABBA0,1) pts(iABBA1,1)], [pts(iABBA0,2) pts(iABBA1,2)], "k-", "LineWidth",1.5, "HandleVisibility","off");
        plot([pts(iABCD0,1) pts(iABBA0,1)], [pts(iABCD0,2) pts(iABBA0,2)], "k-", "LineWidth",1.5, "HandleVisibility","off");
        plot([pts(iABCD1,1) pts(iABBA1,1)], [pts(iABCD1,2) pts(iABBA1,2)], "k-", "LineWidth",1.5, "HandleVisibility","off");
        xlabel("Pitch axis"); ylabel("Rule axis");
        title("Late-window rectangle geometry");
        legend("Location","best"); grid on; axis tight; hold off
    end

    % NEW: leakage AUC
    if opt.plotLeakageAUC
        figure; hold on
        if opt.showPips, draw_pips(pipStarts, pipDur); end

        shaded_sem(t, Out.aucPitch_onPitch_mean, Out.aucPitch_onPitch_sem);
        shaded_sem(t, Out.aucRule_onRule_mean,  Out.aucRule_onRule_sem);
        shaded_sem(t, Out.aucPitch_onRule_mean, Out.aucPitch_onRule_sem);
        shaded_sem(t, Out.aucRule_onPitch_mean, Out.aucRule_onPitch_sem);

        yline(0.5,'k--',"HandleVisibility","off");
        xlabel("Time (s)");
        ylabel("AUC");
        title("1D AUC on each axis (solid=on-axis, cross=leakage)");
        legend("Pitch on PitchAxis","Rule on RuleAxis","Pitch on RuleAxis (leak)","Rule on PitchAxis (leak)", ...
            "Location","best");
        grid on
        ylim([0.45 0.70])  % adjust
        hold off
    end
end

end

% ================= helpers =================

function Xfeat = features_window(X, idxT)
mu = squeeze(mean(X(idxT,:,:), 1, "omitnan")); % units x trials
Xfeat = mu.';                                  % trials x units
end

function [Xb, yb] = balance_binary(X, y)
y = y(:);
i0 = find(y==0); i1 = find(y==1);
n = min(numel(i0), numel(i1));
i0 = i0(randperm(numel(i0), n));
i1 = i1(randperm(numel(i1), n));
idx = [i0; i1];
idx = idx(randperm(numel(idx)));
Xb = X(idx,:);
yb = y(idx);
end

function w = lda_axis(X, y, mode, lambda, sdFloor)
X = double(X); y = y(:);
X0 = X(y==0,:); X1 = X(y==1,:);
mu0 = mean(X0,1); mu1 = mean(X1,1);
dmu = (mu1 - mu0).';

n0 = size(X0,1); n1 = size(X1,1);
if n0 < 2 || n1 < 2, w = dmu; return; end

C0 = cov(X0, 1);
C1 = cov(X1, 1);
Sw = ((n0)*C0 + (n1)*C1) / (n0 + n1);

d = size(Sw,1);
switch mode
    case "ridge"
        sc = trace(Sw)/max(d,1);
        if ~isfinite(sc) || sc < sdFloor^2, sc = sdFloor^2; end
        Swr = Sw + (lambda*sc)*eye(d);
        w = Swr \ dmu;
    case "diag"
        v = diag(Sw);
        v = max(v, sdFloor^2);
        sc = mean(v);
        v = v + lambda*sc;
        w = dmu ./ v;
    otherwise
        error('ldaMode must be "ridge" or "diag".');
end
end

function theta = angle_between(w1, w2, useAbs)
w1 = w1(:); w2 = w2(:);
c = (w1.'*w2) / (max(norm(w1),eps) * max(norm(w2),eps));
c = max(-1,min(1,c));
if useAbs, c = abs(c); end
theta = acosd(c);
end

function idxCell = make_time_windows(t, winSec, mode)
idxCell = cell(1, numel(t));
if winSec <= 0
    for ti = 1:numel(t), idxCell{ti} = ti; end
    return
end
dt = mean(diff(t));
fullBins = max(1, round(winSec/dt));
halfBins = max(1, round((winSec/dt)/2));
mode = lower(string(mode));
for ti = 1:numel(t)
    switch mode
        case "centered"
            a = max(1, ti - halfBins);
            b = min(numel(t), ti + halfBins);
        case "causal"
            a = max(1, ti - fullBins + 1);
            b = ti;
        otherwise
            error('integrateMode must be "causal" or "centered".');
    end
    idxCell{ti} = a:b;
end
end

function draw_pips(pipStarts, pipDur)
yl = ylim;
for k = 1:numel(pipStarts)
    xs = pipStarts(k);
    patch([xs xs+pipDur xs+pipDur xs], [yl(1) yl(1) yl(2) yl(2)], ...
        [0.3 0.3 0.3], "FaceAlpha",0.08, "EdgeColor","none", "HandleVisibility","off");
end
end

function shaded_sem(x, m, s)
x = x(:); m = m(:); s = s(:);
u = m + s;
l = m - s;
patch([x; flipud(x)], [l; flipud(u)], 0.2*[1 1 1], ...
    "FaceAlpha", 0.12, "EdgeColor", "none", "HandleVisibility","off");
plot(x, m, "LineWidth", 2);
end

function [m, sem] = mean_sem(X)
m = mean(X,1,"omitnan");
sem = std(X,0,1,"omitnan") ./ sqrt(max(1,sum(isfinite(X),1)));
end

function auc = auc_rank(y01, score)
% y01 in {0,1}, AUC via rank statistic.
y01 = y01(:); score = score(:);
pos = (y01==1); neg = (y01==0);
nP = sum(pos); nN = sum(neg);
if nP < 2 || nN < 2, auc = NaN; return; end

[scoreS, ord] = sort(score);
r = zeros(size(score));
r(ord) = (1:numel(score)).';

% average ranks for ties
i = 1;
while i <= numel(scoreS)
    j = i;
    while j < numel(scoreS) && scoreS(j+1)==scoreS(i), j=j+1; end
    if j > i
        avg = mean(r(ord(i:j)));
        r(ord(i:j)) = avg;
    end
    i = j+1;
end

sumRanksPos = sum(r(pos));
U = sumRanksPos - nP*(nP+1)/2;
auc = U / (nP*nN);
end
