% run it within DoseResponseCurve_4.m !

c = concs(:);
n = viabilities(:);

nbot0 = max(min(min(n), 0.95), 0);	% clamping to [0, 0.95]
IC50_0 = median(c);

logit = @(x) log(x./(1 - x));
ilogit = @(y) 1./(1 + exp(-y));

theta0 = [log(IC50_0); logit(min(max(nbot0, 1e-6), 1 - 1e-6))];

model = @(c, theta) ilogit(theta(2)) + 1./(1 + c./exp(theta(1)));

obj = @(theta) sum( (model(c,theta) - n).^2 );

opts = optimset('Display','off', 'TolX',1e-10, 'TolFun',1e-10, 'MaxFunEvals',5e4, 'MaxIter',5e4);
theta_hat = fminsearch(obj, theta0, opts);

IC50 = exp(theta_hat(1));
nbot = ilogit(theta_hat(2));

n_hat = model(c, theta_hat);
SS_res = sum((n - n_hat).^2);
SS_tot = sum((n - mean(n)).^2);
R2 = 1 - SS_res/SS_tot;
fprintf('Fit complete: IC50 = %.6g, nbot = %.6g, R^2 = %.4f\n', IC50, nbot, R2);

% --- Plot ---
% Build smooth curve on log grid (avoid zero)
cmin = min(c(c>0));
if isempty(cmin), cmin = 1e-6; end
cmax = max(c);
cgrid = logspace(log10(cmin), log10(max(cmax, cmin*10)), 200);

figure;
hold on
semilogx(c, n, 'o', 'MarkerSize', 6, 'LineWidth', 1.5); hold on;
semilogx(cgrid, model(cgrid, theta_hat), '-', 'LineWidth', 2);
hold off
xlabel('Concentration, c');
ylabel('Viability, n');
title(sprintf('Viability fit: IC_{50}=%.3g, n_{bot}=%.3g, R^2=%.3f', IC50, nbot, R2));
legend({'Data','Fit'}, 'Location','best');
set(gca, 'XScale', 'log');
figname = 'Viability_fit_curve.png';
print(gcf, figname, '-dpng', '-r600'); close;

% --- Save params (same names your downstream script expects) ---
save('DoseResponseCurve_params.mat', 'IC50', 'nbot');
