a_tilde = 17^3;
x_bar = 0.067;

Pss_X = @(ctp) airy(a_tilde^(1/3) * (ctp - x_bar));
normconst = 1/integral(Pss_X, 0, 1);
Pss_X = @(ctp) normconst * Pss_X(ctp);
Nss = integral(Pss_X,0,1);

mu = 0.000036;
a = mu * a_tilde;
xi = a*x_bar;
r_x = 0.1;

b = 0.09;
sigma = sqrt(b);
tau = 7;

ds = 0.01;
s_left = -50;
s_right = 50;
s = s_left:ds:s_right;

dt = 0.1;
t_left = 0;
t_right = 90;
t = t_left:dt:t_right;

s0_left = s_left/5;
s0_right = s_right/5;
s0 = s0_left:ds:s0_right;

c_p = 1;

r_0 = 0.08;
s_r = 2.5;
alpha = 1;
r_fun = @(s) r_0 ./ (1 + exp(-alpha*(s - s_r)));

kmax = 1;
c_50 = 0.05;
gamma = 4;
s_k = 0;
k_0_p = kmax / (1 + c_50/c_p);
k_fun_p = @(s) k_0_p ./ (1 + exp(gamma*(s - s_k)));

h = 1;
c_v = 0.05;
beta = 0.5;
vmax = 10;
s_v = 0;
v_0_p = vmax * c_p^h / (c_v + c_p^h);
v_fun_p = @(s) v_0_p ./ (1 + exp(beta*(s - s_v)));

num_s = numel(s);
num_s0 = numel(s0);
num_t = numel(t);

parpool(128, 'IdleTimeout', Inf);

m = 0;

P_ctp = zeros(num_s0, num_t, num_s);

parfor i = 1:num_s0
	s_0 = s0(i);
	eqn_ctp = @(s,t,P,dPds) pdefun_ctp(s,t,P,dPds,b,s_0,sigma,r_fun,k_fun_p,v_fun_p);
	ic = @(s) icfun(s,b,s_0,sigma);
	bc = @(sl,Pl,sr,Pr,t) bcfun(sl,Pl,sr,Pr,t);
	sol_adv = pdepe(m,eqn_ctp,ic,bc,s,t);
	P_ctp_temp = sol_adv(:,:,1);

	for j = 1:num_s
		for k = 1:num_t
			if P_ctp_temp(k,j) <= 0
				P_ctp_temp(k:end,j) = 0;
				break;
			end
		end
	end

	P_ctp(i,:,:) = P_ctp_temp;
end

x = zeros(size(s0));
[~, idxtau] = min(abs(t - tau));

parfor i = 1:num_s0
	x(i) = trapz(s, squeeze(P_ctp(i,idxtau,:)));
end

figure;
plot(x, s0, 'LineWidth', 2)
xlabel('x')
xlim([0 1])
ylabel('s_0')
title('Plot of s_0(x)')
figname = 'plot_s0_x.png';
print(gcf, figname, '-dpng')
close

clear P_ctp

dx1 = 0.001;
x1 = 0:dx1:1;
s01 = interp1(x, s0, x1, 'linear', 'extrap');

num_x1 = numel(x1);
num_s01 = numel(s01);

num_smallNans = 0;
idx_smallNans = [ ];
num_largeNans = 0;
idx_largeNans = [ ];

for i = 1:num_s01
	if isnan(s01(i))
		num_smallNans = num_smallNans + 1;
		idx_smallNans = [idx_smallNans, i];
	else
		break;
	end
end

for i = num_s01:-1:1
	if isnan(s01(i))
		num_largeNans = num_largeNans + 1;
		idx_largeNans = [idx_largeNans, i];
	else
		break;
	end
end

if num_smallNans > 0
	idx_first_nonNan = idx_smallNans(end) + 1;
	first_nonNan = s01(idx_first_nonNan);
	s01(1:idx_first_nonNan) = linspace(s0_left, first_nonNan, num_smallNans+1);
end
if num_largeNans > 0
	idx_last_nonNan = idx_largeNans(end) - 1;
	last_nonNan = s01(idx_last_nonNan);
	s01(1:idx_last_nonNan) = linspace(last_nonNan, s0_right, num_largeNans+1);
end

s01(1) = s0_left;
s01(2) = (s0_left + s01(3))/2;

s01_col = s01';
x1_col = x1';
table_s0x = table(s01_col, x1_col, 'VariableNames', {'s_0', 'x'});
writetable(table_s0x,'s0_x_standard.txt');

clear P_ctp

txTotalTime = t_right;
num_Holidays = 2;
num_Doses = num_Holidays + 1;
%duration_Holiday = 30;
duration_Holiday = txTotalTime/(num_Doses + num_Holidays);
TotalExposureTime = txTotalTime - duration_Holiday*num_Holidays;
%num_Doses = num_Holidays + 1;
SingleExposureTime = TotalExposureTime / num_Doses;
t_dosing = zeros(1, num_Doses);
for i = 1:numel(t_dosing)
	t_dosing(i) = (i - 1) * (SingleExposureTime + duration_Holiday);
end

t_startHoliday = zeros(1,num_Holidays);
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

c_ref = c_p;
AUC_tot = c_ref * txTotalTime;
c_dose = (AUC_tot / num_Doses) / SingleExposureTime;

k_0 = kmax / (1 + c_50/c_dose);
k_fun = @(s) k_0 ./ (1 + exp(gamma*(s - s_k)));

v_0 = vmax * c_dose^h / (c_v + c_dose^h);
v_fun = @(s) v_0 ./ (1 + exp(beta*(s - s_v)));

x = x1;
s0_x = s01;

desired_num_t2 = 1000;
num_dt2 = floor(desired_num_t2/(numel(timepoints)-1));
t2_stack = [ ];
for i = 1:numel(timepoints)-1
	t2_stack = [t2_stack; linspace(timepoints(i), timepoints(i+1), num_dt2+1)];
end

t2 = [ ];
for i = 1:size(t2_stack,1)
	t2 = [t2, t2_stack(i,:)];
end
t2 = unique(t2);
dt2 = t2(2) - t2(1);

disp(t2);	% for DEBUGGING

num_x = numel(x);
num_s = numel(s);
num_t2 = numel(t2);

%x_interest = [0.01:0.01:0.1, 0.25, 0.5:0.05:0.95];
x_interest = [0.025, 0.05, 0.1, 0.25, 0.5, 0.96];
idx_x_interest = zeros(size(x_interest));
legend_x_interest = cell(size(x_interest));

for i = 1:numel(x_interest)
	[~, idx] = min(abs(x - x_interest(i)));
	idx_x_interest(i) = idx;
	legend_x_interest{i} = ['$x = $' num2str(x_interest(i))];
end

Px_Naive = Pss_X(x);
totalP_t0 = trapz(x, Px_Naive);
Px_Naive = 1/totalP_t0 * Px_Naive;

Pss_S = @(si,s0i) 1/sqrt(2*pi*sigma^2 / b) * exp(-b*(si - s0i)^2 / (2*sigma^2));

P = zeros(num_x, num_t2, num_s);

for i = 1:num_x

	s_0 = s0_x(i);
	N_x = Px_Naive(i);
	
	for j = 1:num_s

		s_j = s(j);
		P(i,1,j) = N_x*Pss_S(s_j,s_0);

	end
end

P0 = squeeze(P(:,1,:));

DoseNumber = 0;

idx_t2stack = 1;
idx_t_begin = 1;

m = 0;
while DoseNumber <= num_Doses

	% drug ON
	t_drugON = t2_stack(idx_t2stack,:);
	num_t_drugON = numel(t_drugON);
	ss = s;
	xx = x;

	idx_t_end = idx_t_begin + num_t_drugON - 1;

	P_init = squeeze(P(:,idx_t_begin,:));

	parfor i = 1:num_x
		s_0 = s0_x(i);
		eqn_drugON = @(s,t,P,dPds) pdefun_drug(s,t,P,dPds,b,s_0,sigma,r_fun,k_fun,v_fun);
		ic = @(s) interp1(ss,P_init(i,:),s,'linear', 'extrap');
		bc = @(sl,Pl,sr,Pr,t) bcfun(sl,Pl,sr,Pr,t);
		sol_drugON = pdepe(m,eqn_drugON, ic, bc, s, t_drugON);
		Pxst_drugON_temp = sol_drugON(:,:,1);

		for j = 1:num_s
			for k = 1:num_t_drugON

				if Pxst_drugON_temp(k,j) <= 0
					Pxst_drugON_temp(k:end,j) = 0;
					break;
				end
			end
		end

		Pxst_drugON(i,:,:) = Pxst_drugON_temp;

	end

	P(:,idx_t_begin:idx_t_end,:) = Pxst_drugON;

	DoseNumber = DoseNumber + 1;
	if DoseNumber == num_Doses
		break;
	end

	idx_t2stack = idx_t2stack + 1;

	% drug OFF - Holiday 

	% x dynamics
	Px_postDrug = zeros(size(x));
	for i = 1:num_x
		Px_postDrug(i) = trapz(s, squeeze(Pxst_drugON(i,end,:)));
	end

	idx_t_begin = idx_t_end;
	t_holiday = t2_stack(idx_t2stack,:);
	num_t_holiday = numel(t_holiday);
	idx_t_end = idx_t_begin + num_t_holiday - 1;

	Px_init = Px_postDrug;
	Px_current = Px_init;
	t_current = t_holiday(1);
	%dt = t_holiday(2) - t_holiday(1);

	Pxt_holiday = zeros(num_t_holiday, num_x);
	Pxt_holiday(1,:) = Px_init;

	for i = 1:num_t_holiday-1
		Nt = trapz(x, Px_current);
		tspan = [t_holiday(i), (t_holiday(i) + t_holiday(i+1))/2, t_holiday(i+1)];
		eqn_x = @(x,t,Px,dPxdx) pdefun_Px(x,t,Px,dPxdx,xi,a,mu,r_x,Nt,Nss);
		ic_x = @(x) interp1(xx, Px_current, x, 'linear', 'extrap');
		bc_x = @(xl,Pxl,xr,Pxr,t) bcfun_Px(xl,Pxl,xr,Pxr,t);
		sol_current = pdepe(m,eqn_x,ic_x,bc_x,x,tspan);

		Px_new = sol_current(end,:,1);
		Pxt_holiday(i+1,:) = Px_new;

		Px_current = Px_new;
	end

	%eqn_x = @(x,t,Px,dPxdx) pdefun_Px(x,t,Px,dPxdx,xi,a,mu);
	%ic_x = @(x) interp1(xx, Px_postDrug, x, 'linear', 'extrap');
	%bc_x = @(xl,Pxl,xr,Pxr,t) bcfun_Px(xl,Pxl,xr,Pxr,t);
	%sol_x = pdepe(m,eqn_x,ic_x,bc_x,x,t_holiday);
	%Pxt_holiday = sol_x(:,:,1);

	Pst_holiday = zeros(num_x,num_t_holiday,num_s);

	parfor i = 1:num_x
		s_0 = s0_x(i);
		eqn_nodrug = @(s,t,P,dPds) pdefun_nodrug(s,t,P,dPds,b,s_0,sigma);
		Ps_postDrug = squeeze(Pxst_drugON(i,end,:));
		ic_nodrug = @(s) interp1(ss, Ps_postDrug, s, 'linear', 'extrap');
		bc_nodrug = @(sl,Pl,sr,Pr,t) bcfun(sl,Pl,sr,Pr,t);
		sol_nodrug = pdepe(m,eqn_nodrug,ic_nodrug,bc_nodrug,s,t_holiday);

		Pst_holiday(i,:,:) = sol_nodrug(:,:,1);
	end

	for i = 1:num_x
		Px_0 = Pxt_holiday(1,i);
		
		for j = idx_t_begin:idx_t_end
			Px_j = Pxt_holiday(j-idx_t_begin+1,i);

			for k = 1:num_s
		
				P(i,j,k) = Pst_holiday(i,j-idx_t_begin+1,k) * Px_j/Px_0;

			end
		end
	end

	idx_t_begin = idx_t_end;
	idx_t2stack = idx_t2stack + 1;

end

A = zeros(num_x, num_t2);	% survival curves for each x-clone

for i = 1:num_x
	for j = 1:num_t2
		A(i,j) = trapz(s, squeeze(P(i,j,:)));
	end
end

figure;
for i = 1:numel(x_interest)
	idx_x = idx_x_interest(i);
	semilogy(t2, A(idx_x,:)/A(idx_x,1), 'linewidth', 1.5)
	hold on
end
hold off
xlabel('Time', 'fontsize', 14)
ylabel('Survival', 'fontsize', 14)
legend(legend_x_interest, 'interpreter', 'latex')
figname = 'survivalcurves_holiday.png';
print(gcf, figname, '-dpng')
close

P_tot = zeros(size(t2));
for i = 1:num_t2
	integral_ds = zeros(size(x));

	for j = 1:num_x
		integral_ds(j) = trapz(s, squeeze(P(j,i,:)));
	end

	P_tot(i) = trapz(x, integral_ds);
end

Px = zeros(num_x, num_t2);
parfor i = 1:num_x
	for j = 1:num_t2
		Px(i,j) = trapz(s, squeeze(P(i,j,:)));
	end
end

idx_t2imepoints = zeros(size(timepoints));
for i = 1:numel(timepoints)
	[~, idx] = min(abs(t2 - timepoints(i)));
	idx_t2imepoints(i) = idx;
end

idx_timepoints = zeros(size(timepoints));
for i = 1:numel(timepoints)
	[~, idx] = min(abs(t - timepoints(i)));
	idx_timepoints(i) = idx;
end

figure;
hold on
for i = 1:numel(timepoints)
	idx_t = idx_t2imepoints(i);
	
	norm_x = 1/trapz(x, squeeze(Px(:,idx_t)));

	plot(x, norm_x*Px(:,idx_t), 'linewidth', 1.5)
end
hold off
xlabel('$x$', 'interpreter', 'latex', 'fontsize', 16)
ylabel('Normalized Distribution', 'fontsize', 14)
legend(legend_timepoints)
figname = 'Px_timepoints.png';
print(gcf, figname, '-dpng')
close 



% now to compare with no-holiday continuous treatment
k_0_nohol = kmax / (1 + c_50/c_ref);
k_fun_nohol = @(s) k_0_nohol ./ (1 + exp(gamma*(s - s_k)));

v_0_nohol = vmax*c_ref^h/(c_v + c_ref^h);
v_fun_nohol = @(s) v_0_nohol ./ (1 + exp(beta*(s - s_v)));

P_nohol = zeros(num_x, num_t, num_s);

parfor i = 1:num_x
	s_0 = s0_x(i);
	eqn_nohol = @(s,t,P,dPds) pdefun_drug(s,t,P,dPds,b,s_0,sigma,r_fun,k_fun_nohol,v_fun_nohol);
	ic_nohol = @(s) icfun(s,b,s_0,sigma);
	bc_nohol = @(sl,Pl,sr,Pr,t) bcfun(sl,Pl,sr,Pr,t);
	sol_nohol = pdepe(m,eqn_nohol,ic_nohol,bc_nohol,s,t);
	P_nohol_temp = sol_nohol(:,:,1);

	for j = 1:num_s
		for k = 1:num_t

			if P_nohol_temp(k,j) <= 0
				P_nohol_temp(k:end,j) = 0;
				break;
			end
		end
	end

	P_nohol(i,:,:) = P_nohol_temp;

end

A_nohol = zeros(num_x, num_t);

for i = 1:num_x
	for j = 1:num_t
		A_nohol(i,j) = trapz(s, squeeze(P_nohol(i,j,:)));
	end
end

figure;
for i = 1:numel(x_interest)
	idx_x = idx_x_interest(i);

	semilogy(t, A_nohol(idx_x,:), 'linewidth', 1.5)
	hold on
end
hold off
xlabel('Time', 'fontsize', 14)
ylabel('Survival', 'fontsize', 14)
legend(legend_x_interest, 'interpreter', 'latex')
figname = 'survivalcurves_noholiday.png';
print(gcf, figname, '-dpng')
close

P_tot_nohol = zeros(size(t));
weights = Pss_X(x);
for i = 1:num_t
	integral_ds_nohol = zeros(size(x));

	for j = 1:num_x
		integral_ds_nohol(j) = weights(j)*trapz(s, squeeze(P_nohol(j,i,:)));
	end

	P_tot_nohol(i) = trapz(x, integral_ds_nohol);
end

figure;
semilogy(t, P_tot_nohol/P_tot_nohol(1), 'linewidth', 2)
hold on
semilogy(t2, P_tot/P_tot(1), 'linewidth', 2)
hold off
xlabel('Time', 'fontsize', 14)
ylabel('Survival', 'fontsize', 14)
legend({'Continuous', 'Holidays'})
figname = 'TotalPopulation.png';
print(gcf, figname, '-dpng')
close

Px_nohol_end = zeros(size(x));
for i = 1:num_x
	x_i = x(i);
	N_x_i = Pss_X(x_i);
	Px_nohol_end(i) = N_x_i * trapz(s, squeeze(P_nohol(i,end,:)));
end
normconst = 1/trapz(x, Px_nohol_end);
Px_nohol_end = normconst * Px_nohol_end;

legend_all_last = 'Continuous';
legend_all = [legend_timepoints, legend_all_last];

figure;
hold on
for i = 1:numel(timepoints)
	idx_t = idx_t2imepoints(i);
	
	norm_x = 1/trapz(x, squeeze(Px(:,idx_t)));

	plot(x, norm_x*Px(:,idx_t), 'linewidth', 1.5)
end
plot(x, Px_nohol_end, '--', 'linewidth', 1.5, 'color', 'red')
hold off
xlabel('$x$', 'interpreter', 'latex', 'fontsize', 16)
ylabel('Normalized Distribution', 'fontsize', 14)
legend(legend_all, 'interpreter', 'latex')
figname = 'Px_timepoints_together.png';
print(gcf, figname, '-dpng')
close

save('FinalData.mat');



%% FUNCTIONS

function[g,f,h] = pdefun_ctp(s,t,P,dPds,b,s_0,sigma,r_fun,k_fun_p,v_fun_p)

	r = r_fun(s);
	k = k_fun_p(s);
	v = v_fun_p(s);

	g = 1;
	f = sigma^2 * dPds + b*(s - s_0 - v)*P;
	h = (r - k)*P;

end

function[g,f,h] = pdefun_drug(s,t,P,dPds,b,s_0,sigma,r_fun,k_fun,v_fun)

	r = r_fun(s);
	k = k_fun(s);
	v = v_fun(s);

	g = 1;
	f = sigma^2 * dPds + b*(s - s_0 - v)*P;
	h = (r - k)*P;

end

function[g,f,h] = pdefun_nodrug(s,t,P,dPds,b,s_0,sigma)

	g = 1;
	f = sigma^2 * dPds + b*(s - s_0)*P;
	h = 0;

end

%%% Initial condition
function P0 = icfun(s,b,s_0,sigma)

	P0 = 1/sqrt(2*pi*sigma^2 / b) * exp(-b*(s - s_0)^2 / (2*sigma^2));

end

%%% Boundary Condition
function[pl, ql, pr, qr] = bcfun(sl,Pl,sr,Pr,t)

	pl = 0;
	ql = 1;
	pr = 0;
	qr = 1;

end 


function[g,f,h] = pdefun_Px(x,t,Px,dPxdx,xi,a,mu,r_x,Nt,Nss)

	g = 1;
	f = mu * dPxdx;
	h = (xi - a*x)*Px + r_x*Px*(1 - Nt/Nss);

end

function[pl,ql,pr,qr] = bcfun_Px(xl,Pxl,xr,Pxr,t)

	pl = 0;
	ql = 1;
	pr = 0;
	qr = 1;

end

	
