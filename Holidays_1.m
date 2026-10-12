%%%%%%!!! After running InitialAnalysis_8.m !!!%%%%%%%

load('Params.mat');

x = x(:);
s0 = s0(:);

ds = 0.01;
s_left = -50;
s_right = 50;
s = (s_left:ds:s_right).';

num_s = numel(s);
num_x = numel(x);
dx = x(2) - x(1);

t_left = 0;
t_right = 100;
dt = 0.1;
t = (t_left:dt:t_right).';

%num_t = numel(t);

c_ref = 1;	% drug dose (continuous AUC-equivalent)

Pss_X_raw = @(ctp) airy(a_tilde^(1/3) * (ctp - x_bar));
normconst = 1/integral(Pss_X_raw, 0, 1);
Pss_X = @(ctp) normconst * Pss_X_raw(ctp);
Px = Pss_X(x);
Px = Px(:);

Nss = 1;
N_init = Nss;

% Holiday Scheduling 

txTotalTime = t_right;
num_Holidays = 2;
num_Doses = num_Holidays + 1;

duration_Holiday = txTotalTime/(num_Doses + num_Holidays);	% equal interval 
TotalExposureTime = txTotalTime - duration_Holiday*num_Holidays;
SingleExposureTime = TotalExposureTime / num_Doses;

t_dosing = zeros(1, num_Doses);
for i = 1:numel(t_dosing)

	t_dosing(i) = (i-1) * (SingleExposureTime + duration_Holiday);

end

t_startHoliday = zeros(1, num_Holidays);
for i = 2:num_Doses

	t_startHoliday(i-1) = t_dosing(i) - duration_Holiday;

end

timepoints = [t_startHoliday, t_dosing, t_dosing(end)+SingleExposureTime];
timepoints = sort(timepoints);
legend_timepoints = cell(size(timepoints));
for i = 1:numel(timepoints)

	legend_timepoints{i} = ['$t = $' num2str(timepoints(i))];

end

t_OneCycle = SingleExposureTime + duration_Holiday;

AUC_tot = c_ref * txTotalTime;
c_dose = (AUC_tot / num_Doses) / SingleExposureTime;
%c_dose = 1;	% if you want to manually specify the dose for each drug-on phase

Rs_base = r_0 ./ (1 + exp(-alpha*(s - s_r)));

k_0 = kmax / (1 + c_50/c_dose);
k_grid = k_0 ./ (1 + exp(gamma*(s - s_k)));

v_0 = vmax*c_dose^h/(c_v + c_dose^h);
v_grid = v_0 ./ (1 + exp(beta*(s - s_v)));

x_interest = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75];
idx_x_interest = zeros(size(x_interest));
legend_x_interest = cell(size(x_interest));

for i = 1:numel(x_interest)

	[~, idx] = min(abs(x - x_interest(i)));
	idx_x_interest(i) = idx;
	legend_x_interest{i} = ['$x = $' num2str(x_interest(i))];

end

desired_num_t2 = 5000;
num_dt2 = floor(desired_num_t2/(numel(timepoints)-1));
t2_stack = zeros(numel(timepoints)-1, num_dt2+1);
for i = 1:numel(timepoints)-1

	t2_stack(i,:) = linspace(timepoints(i), timepoints(i+1), num_dt2+1);

end

t2 = t2_stack(1,:).';
for seg = 2:size(t2_stack, 1)

	t2 = [t2; t2_stack(seg,2:end).'];

end
num_t2 = numel(t2);
dt2 = t2(2) - t2(1);

Px_Naive = Px;

icfun = @(sgrid,s0x) 1/sqrt(2*pi*sigma^2 / b) * exp( (-b/(2*sigma^2)) * (sgrid - s0x).^2 );
P_Naive = zeros(num_s, num_x);

for j = 1:num_x

	P_Naive(:,j) = Px(j)*icfun(s,s0(j));

end

N0 = trapz(x, trapz(s, P_Naive, 1));
P_Naive = (N_init/N0)*P_Naive;	% normalzied to N_init 
Ps_Naive = trapz(x, P_Naive, 2);
Ps_Naive = Ps_Naive(:);
Ps_Naive = Ps_Naive / max(trapz(s, Ps_Naive), eps);

uno = ones(num_s, 1);
Lap = spdiags([uno -2*uno uno], [-1 0 1], num_s, num_s)/(ds^2);
Lap(1,1) = -2/(ds^2);
Lap(1,2) = 2/(ds^2);
Lap(end,end) = -2/(ds^2);
Lap(end,end-1) = 2/(ds^2);

I = speye(num_s);
AhalfL = I - 0.5*dt2*sigma^2*Lap;
AhalfR = I + 0.5*dt2*sigma^2*Lap;

[Lfac, Ufac, Pperm] = lu(AhalfL);
solve_half_diff = @(Y) Ufac \ (Lfac \ (Pperm*(AhalfR*Y)));

S = repmat(s, 1, num_x);
S0 = repmat(s0(:).', num_s, 1);
Vs_drug = -b*(S - S0 - repmat(v_grid, 1, num_x));	% for drug on
Vs_hol = -b*(S - S0);	% for drug off

DoseNumber = 0;
idx_t2stack = 1;
idx_t_begin = 1;

cycleNumber = 1:num_Doses;

P = zeros(num_s, num_x, num_t2);
P(:,:,1) = P_Naive;

% operators for x (during holiday)
uno_x = ones(num_x, 1);
Lapx = spdiags([uno_x -2*uno_x uno_x], [-1 0 1], num_x, num_x)/(dx^2);
Lapx(1,1) = -2/(dx^2);
Lapx(1,2) = 2/(dx^2);
Lapx(end,end) = -2/(dx^2);
Lapx(end,end-1) = 2/(dx^2);

Ix = speye(num_x);
BxL = Ix - 0.5*dt2*mu*Lapx;
BxR = Ix + 0.5*dt2*mu*Lapx;
[Lx, Ux, Pxperm] = lu(BxL);
solve_half_diff_x = @(y) Ux \ (Lx \ (Pxperm*(BxR*y)));

w_Rx_on = 1/(1 + c_dose/c_R);
w_Rs_on = 1/(1 + c_R/c_dose);
w_Rx_off = 1;
w_Rs_off = 0;

tolT = 1e-12;

idx_t = 1;

heatmapON = 0;	% 1 if you want to plot them, 0 if you don't
plotevery = 10;

for seg = 1:size(t2_stack, 1)

	tseg = t2_stack(seg,:).';	% (num_d2+1) x 1
	nseg = numel(tseg);
	t0 = tseg(1);

	isDrugOn = any(abs(t0 - t_dosing) < tolT);	% if it's not drug-on then starts at t_startHoliday

	for i = 1:nseg-1

		Pnow = P(:,:,idx_t);
		N_now = trapz(x, trapz(s, Pnow, 1));
		g = 1 - N_now/Nss;

		Ptmp = solve_half_diff(Pnow);	% half diffiusion in s

		if isDrugOn
			vS = repmat(v_grid, 1, num_x);	% drug-on
			u = -b*(S - S0 - vS);
		else
			u = -b*(S - S0);	% holiday (drug-off)
		end

		Ptmp = SL_advect_2D(Ptmp, s, dt2, u);	% advect in s

		du = deriv_2D(u,ds);
		
		if isDrugOn
			Rx_x = xi - a*x + r_x*g;
			RxS = repmat(Rx_x(:).', num_s, 1);
			RsS = repmat(Rs_base*g, 1, num_x);

			r_grid = w_Rx_on*RxS + w_Rs_on*RsS;
			kS = repmat(k_grid, 1, num_x);

			net = (r_grid - kS) - du;
		else
			% holiday 
			net = -du;
		end

		Ptmp = Ptmp .* exp(net*dt2);	% birth and death
		Ptmp = solve_half_diff(Ptmp);	% half diffusion in s
		Ptmp = max(Ptmp,0);

		if ~isDrugOn	% holiday: diffusion in x and Rx growth
			px = trapz(s, Ptmp, 1).';
			Nt_x = trapz(x, px);

			px_half = solve_half_diff_x(px);
			
			g_x = xi - a*x + r_x*(1 - Nt_x/Nss);
			px_react = px_half .* exp(dt2*g_x);
			px_react(px_react < 0) = 0;

			px_target = solve_half_diff_x(px_react);
			px_target(px_target < 0) = 0;

			px_current = trapz(s, Ptmp, 1).';
			scale = px_target ./ max(px_current, 1e-300);

			Pnew = Ptmp .* repmat(scale.', num_s, 1);
		else
			Pnew = Ptmp;
		end

		Pnew = max(Pnew, 0);

		idx_t = idx_t + 1;
		P(:,:,idx_t) = Pnew; 

		if heatmapON == 1 && mod(idx_t-1, plotevery) == 0
			figure;
			imagesc(x, s, P(:,:,idx_t))
			set(gca, 'YDir', 'normal')
			axis tight
			colormap(jet)
			colorbar
			xlabel('$x$', 'interpreter', 'latex')
			ylabel('$s$', 'interpreter', 'latex')
			xlim([0 0.45])
			ylim([-7 7])
			ax = gca;
			ax.FontSize = 14;
			ax.XLabel.FontSize = 18;
			ax.YLabel.FontSize = 18;
			figname = ['P_xs_t' num2str(idx_t) '.png'];
			print(gcf, figname, '-dpng', '-r600')
			close
		end

	end

end

% making plots 
idx_t2_timepoints = zeros(size(timepoints));
for i = 1:numel(timepoints)

	[~, idx] = min(abs(t2 - timepoints(i)));
	idx_t2_timepoints(i) = idx;

end

A_xt = zeros(num_x, num_t2);
N_t = zeros(num_t2, 1);
Px_t = zeros(num_x, numel(timepoints));
Ps_t = zeros(num_s, numel(timepoints));

Px_mass0 = trapz(s, P(:,:,1), 1).';

for i = 1:num_t2

	Px_mass = trapz(s, P(:,:,i), 1).';
	A_xt(:,i) = Px_mass ./ Px_mass0;
	N_t(i) = trapz(x, Px_mass);

end

for i = 1:numel(timepoints)

	it = idx_t2_timepoints(i);

	Px_snap = trapz(s, P(:,:,it), 1).';
	Px_snap = Px_snap / max(trapz(x, Px_snap), eps);
	Px_t(:,i) = Px_snap;

	Ps_snap = trapz(x, P(:,:,it), 2);
	Ps_snap = Ps_snap / max(trapz(s, Ps_snap), eps);
	Ps_t(:,i) = Ps_snap;

end

save('Holiday_results.mat', 'x', 's0', 's', 'c_ref', 't2', 'A_xt', 'N_t', 'Px_t', 'Ps_t', '-v7.3');

% survival curve for each clone x
figure;
for i = 1:numel(x_interest)
	
	idx = idx_x_interest(i);
	semilogy(t2, A_xt(idx,:), 'linewidth', 1.5)
	hold on

end
hold off
xlabel('Time', 'interpreter', 'latex')
ylabel('Survival Fraction', 'interpreter', 'latex')
ax = gca;
ax.FontSize = 14;
ax.XLabel.FontSize = 18;
ax.YLabel.FontSize = 18;
lgd = legend(legend_x_interest, 'interpreter', 'latex', 'location', 'northwest');
lgd.FontSize = 12;
figname = 'A_xt_holiday.png';
print(gcf, figname, '-dpng', '-r600')
close

% total population
figure;
semilogy(t2, N_t, 'linewidth', 2.5)
xlabel('Time', 'interpreter', 'latex')
ylabel('Total Population', 'interpreter', 'latex')
ax = gca;
ax.FontSize = 14;
ax.XLabel.FontSize = 18;
ax.YLabel.FontSize = 18;
figname = 'TotalPop.png';
print(gcf, figname, '-dpng', '-r600')
close

% Px
figure;
hold on
plot(x, Px_Naive, '--', 'linewidth', 1.5, 'color', 'k')
for i = 2:numel(timepoints)

	plot(x, Px_t(:,i), '-', 'linewidth', 1.5)

end
hold off
xlabel('$x$', 'interpreter', 'latex')
ylabel('$P_x(x)$', 'interpreter', 'latex', 'rotation', 0)
xlim([0 1])
ax = gca;
ax.FontSize = 14;
ax.XLabel.FontSize = 18;
ax.YLabel.FontSize = 18;
lgd = legend(legend_timepoints, 'interpreter', 'latex', 'location', 'northeast');
lgd.FontSize = 12;
figname = 'Px.png';
print(gcf, figname, '-dpng', '-r600')
close

% Ps
figure;
hold on
plot(s, Ps_Naive, '--', 'linewidth', 1.5, 'color', 'k')
for i = 2:numel(timepoints)

	plot(s, Ps_t(:,i), '-', 'linewidth', 1.5)

end
hold off
xlabel('$s$', 'interpreter', 'latex')
ylabel('$P_s(s)$', 'interpreter', 'latex', 'rotation', 0)
xlim([-10 10])
ax = gca;
ax.FontSize = 14;
ax.XLabel.FontSize = 18;
ax.YLabel.FontSize = 18;
lgd = legend(legend_timepoints, 'interpreter', 'latex', 'location', 'northwest');
lgd.FontSize = 12;
figname = 'Ps.png';
print(gcf, figname, '-dpng', '-r600')
close




%% FUNCTIONS

function du = deriv_2D(u, ds)
	
	% u is num_s x num_x
	du = zeros(size(u));
	du(2:end-1,:) = (u(3:end,:) - u(1:end-2,:))/(2*ds);
	du(1,:) = (u(2,:) - u(1,:))/ds;
	du(end,:) = (u(end,:) - u(end-1,:))/ds;

end

function Yadv = SL_advect_2D(Y, s, dt, u)

	% Y and u are num_s x num_x
	[num_s, num_x] = size(Y);
	S = repmat(s, 1, num_x);
	s_from = S - u*dt;
	s_from = max(min(s_from, s(end)), s(1));	% clamp

	idx = (s_from - s(1))/(s(2) - s(1)) + 1;
	i0 = floor(idx);
	i0 = max(i0,1);
	i1 = min(i0+1,num_s);
	w1 = idx - i0;
	w0 = 1 - w1;

	[~, J] = ndgrid(1:num_s, 1:num_x);
	lin = @(I,J) sub2ind([num_s, num_x], I, J);

	Yadv = w0.*Y(lin(i0,J)) + w1.*Y(lin(i1,J));

end
	







