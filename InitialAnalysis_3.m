a_tilde = 17^3;
x_bar = 0.067;

Pss_X = @(ctp) airy(a_tilde^(1/3)*(ctp - x_bar));
A_Pss = 1/Pss_X(0);
Pss_X = @(ctp) A_pss * Pss_X(ctp);

b = 0.09;
sigma = sqrt(b);
tau = 7;

%%% Grids
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

%%% Drug concentration 
c = 1;

%%% proliferation
r_0 = 0.08;
s_r = 2.5;
alpha = 1;

r_fun = @(s) r_0 ./ (1 + exp(-alpha*(s - s_r)));

%%% killing
kmax = 1;
c_50 = 0.05;
gamma = 4;
s_k = 0;
s_k = 0;
k_0 = kmax / (1 + c_50/c);

k_fun = @(s) k_0 ./ (1 + exp(gamma*(s - s_k)));				  

%%% advection 
      
c_v = 0.05;
beta = 0.5;
vmax = 10;
s_v = 0;
v_0 = vmax*c/(c_v + c);
		       
v_fun = @(s) v_0 ./ (1 + exp(beta*(s - s_v)));
			  
					       
g_fun = @(s) r_fun(s) - k_fun(s);

legend_rates = {'r(s)', 'k(s)', 'g(s)', 'v(s)'};

num_s = numel(s);
num_s0 = numel(s0);
num_t = numel(t);

figure;
set(gca, 'visible', 'off')
text(0.5, 0.5, ['kmax = ' num2str(kmax) ', gamma = ' num2str(gamma) ', vmax = ' num2str(vmax) ', beta = ' num2str(beta)], 'FontSize', 14, 'HorizontalAlignment', 'center');
figname = 'kmax_gamma_vmax_beta';
print(gcf, figname, '-dpng')
close

figure;
plot(s, r_fun(s), 'LineWidth', 1.5)
hold on
plot(s, k_fun(s), 'LineWidth', 1.5)
plot(s, g_fun(s), 'LineWidth', 1.5)
plot(s, v_fun(s), 'LineWidth', 1.5)
hold off
xlabel('s')
xlim([-7.5 7.5])
ylabel('Rate')
ylim([-4 4])
legend(legend_rates)
title('Plots of Rates')
figname = 'plot_rates';
print(gcf, figname, '-dpng')
close

parpool(128, 'IdleTimeout', 180);

% Operationally define x 
P_adv = zeros(num_s0, num_t, num_s);
m = 0
parfor i = 1:num_s0
	s_0 = s0(i);
	eqn_adv = @(s,t,P,dPds) pdefun_adv(s,t,P,dPds,b,s_0,sigma,r_fun,k_fun,v_fun);
	ic = @(s) icfun(s,b,s_0,sigma);
	bc = @(sl,Pl,sr,Pr,t) bcfun(sl,Pl,sr,Pr,t);
	sol_adv = pdepe(m,eqn_adv,ic,bc,s,t);
	P_adv_temp = sol_adv(:,:,1);
	for j = 1:num_s
		for k = 1:num_t
			if P_adv_temp(k,j) <= 0
				P_adv_temp(k:end,j) = 0;
				break;
			end
		end
	end
	P_adv(i,:,:) = P_adv_temp;
end

x = zeros(size(s0));
[~, idxtau] = min(abs(t - tau));

for i = 1:num_s0
	x(i) = trapz(s, squeeze(P_adv(i,idxtau,:)));
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

clear P_adv

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

s01_col = s01';
x1_col = x1';
table_s0x = table(s01_col, x1_col, 'VariableNames', {'s_0', 'x'});
writetable(table_s0x,'s0_x_standard.txt');

%% Survival Analysis
P1_adv = zeros(num_s01, num_t, num_s);
m = 0;
parfor i = 1:num_s01
	s_0 = s01(i);
	eqn1_adv = @(s,t,P,dPds) pdefun_adv(s,t,P,dPds,b,s_0,sigma,r_fun,k_fun,v_fun);
	ic = @(s) icfun(s,b,s_0,sigma);
	bc = @(sl,Pl,sr,Pr,t) bcfun(sl,Pl,sr,Pr,t);
	sol1_adv = pdepe(m,eqn1_adv,ic,bc,s,t);
	P1_adv_temp = sol1_adv(:,:,1);
	for j = 1:num_s
		for k = 1:num_t
			if P1_adv_temp(k,j) <= 0
				P1_adv_temp(k:end,j) = 0;
				break;
			end
		end
	end
	P1_adv(i,:,:) = P1_adv_temp;
end

A_adv = zeros(num_s01, num_t);
parfor i = 1:num_x1
	for j = 1:num_t
		A_adv(i,j) = trapz(s, squeeze(P1_adv(i,j,:)));
	end
end

timepoints = [0, 2.5, 5, 10, 20, 30, 60, 90];
idx_timepoints = zeros(size(timepoints));
legend_timepoints = cell(size(timepoints));
for i = 1:numel(timepoints)
	[~, idx] = min(abs(t - timepoints(i)));
	idx_timepoints(i) = idx;
	legend_timepoints{i} = ['t = ' num2str(timepoints(i))];
end

x1_samples1 = 0.01:0.01:0.15;
x1_samples2 = 0.015:0.001:0.025;	% around first transition point
x1_samples3 = 0.125:0.001:0.135;	% around the second transition point
x1_samples4 = [0.01:0.01:0.1, 0.2:0.1:0.9, 0.99];

idx_x1_samples1 = zeros(size(x1_samples1));
idx_x1_samples2 = zeros(size(x1_samples2));
idx_x1_samples3 = zeros(size(x1_samples3));
idx_x1_samples4 = zeros(size(x1_samples4));

legend_x1_samples1 = cell(size(x1_samples1));
legend_x1_samples2 = cell(size(x1_samples2));
legend_x1_samples3 = cell(size(x1_samples3));
legend_x1_samples4 = cell(size(x1_samples4));

for i = 1:numel(x1_samples1)
	[~, idx] = min(abs(x1 - x1_samples1(i)));
	idx_x1_samples1(i) = idx;
	legend_x1_samples1{i} = ['x = ' num2str(x1_samples1(i))];
end
for i = 1:numel(x1_samples2)
	[~, idx] = min(abs(x1 - x1_samples2(i)));
	idx_x1_samples2(i) = idx;
	legend_x1_samples2{i} = ['x = ' num2str(x1_samples2(i))];
end
for i = 1:numel(x1_samples3)
	[~, idx] = min(abs(x1 - x1_samples3(i)));
	idx_x1_samples3(i) = idx;
	legend_x1_samples3{i} = ['x = ' num2str(x1_samples3(i))];
end
for i = 1:numel(x1_samples4)
        [~, idx] = min(abs(x1 - x1_samples4(i)));
        idx_x1_samples4(i) = idx;
        legend_x1_samples4{i} = ['x = ' num2str(x1_samples4(i))];
end

%{
for i = 1:numel(x1_samples1)
	idx_x1 = idx_x1_samples1(i);
	figure;
	for j = idx_timepoints
		plot(s, squeeze(P1_adv(idx_x1,j,:)), 'linewidth', 1)
		hold on
	end
	hold off
	xlabel('s')
	xlim([-5, 7])
	ylabel('P_S(t)')
	legend(legend_timepoints)
	title(['Population Distribution for x = ' num2str(x1_samples1(i))])
	figname = ['PopDistr_samples1_' num2str(i) '.png']
	print(gcf, figname, '-dpng')
	close
end

for i = 1:numel(x1_samples2)
	idx_x1 = idx_x1_samples2(i);
	figure;
	for j = idx_timepoints
		plot(s, squeeze(P1_adv(idx_x1,j,:)), 'linewidth', 1)
		hold on
	end
	hold off
	xlabel('s')
	xlim([-5, 7])
	ylabel('P_S(t)')
	legend(legend_timepoints)
	title(['Population Distribution for x = ' num2str(x1_samples2(i))])
	figname = ['PopDistr_samples2_' num2str(i) '.png']
	print(gcf, figname, '-dpng')
	close
end

for i = 1:numel(x1_samples3)
	idx_x1 = idx_x1_samples3(i);
	figure;
	for j = idx_timepoints
		plot(s, squeeze(P1_adv(idx_x1,j,:)), 'linewidth', 1)
		hold on
	end
	hold off
	xlabel('s')
	xlim([-5, 7])
	ylabel('P_S(t)')
	legend(legend_timepoints)
	title(['Population Distribution for x = ' num2str(x1_samples3(i))])
	figname = ['PopDistr_samples3_' num2str(i) '.png']
	print(gcf, figname, '-dpng')
	close
end

for i = 1:numel(x1_samples4)
	idx_x1 = idx_x1_samples4(i);
	figure;
	for j = idx_timepoints
		plot(s, squeeze(P1_adv(idx_x1,j,:)), 'linewidth', 1)
		hold on
	end
	hold off
	xlabel('s')
	xlim([-5, 7])
	ylabel('P_S(t)')
	legend(legend_timepoints)
	title(['Population Distribution for x = ' num2str(x1_samples4(i))])
	figname = ['PopDistr_samples4_' num2str(i) '.png']
	print(gcf, figname, '-dpng')
	close
end

figure;
for i = 1:numel(x1_samples1)
	idx_x1 = idx_x1_samples1(i);
	semilogy(t, A_adv(idx_x1,:), 'linewidth', 1)
	hold on
end
xlabel('Time')
xlim([0 30])
ylabel('Survival')
ylim([10^(-5) 2*10^(1)])
legend(legend_x1_samples1, 'location', 'eastoutside')
title('Survival Curves')
figname = 'SurvivalCurve_t30_samples1.png';
print(gcf, figname, '-dpng')
xlim([0 60])
figname = 'SurvivalCurve_t60_samples1.png';
print(gcf, figname, '-dpng')
xlim([0 90])
figname = 'SurvivalCurve_t90_samples1.png';
print(gcf, figname, '-dpng')
xlim([0 2.5])
ylim([8*10^(-2) 1.1*10^(0)])
figname = 'SurvivalCurve_short_samples1.png';
print(gcf, figname, '-dpng')
close

figure;
for i = 1:numel(x1_samples2)
	idx_x1 = idx_x1_samples2(i);
	semilogy(t, A_adv(idx_x1,:), 'linewidth', 1)
	hold on
end
	
xlabel('Time')
xlim([0 30])
ylabel('Survival')
ylim([10^(-5) 2*10^(1)])
legend(legend_x1_samples2, 'location', 'eastoutside')
title('Survival Curves')
figname = 'SurvivalCurve_t30_samples2.png';
print(gcf, figname, '-dpng')
xlim([0 60])
figname = 'SurvivalCurve_t60_samples2.png';
print(gcf, figname, '-dpng')
xlim([0 90])
figname = 'SurvivalCurve_t90_samples2.png';
print(gcf, figname, '-dpng')
xlim([0 2.5])
ylim([10^(-1) 1.1*10^(0)])
figname = 'SurvivalCurve_short_samples2.png';                                                                                                                                                         print(gcf, figname, '-dpng')  
close

figure;
for i = 1:numel(x1_samples3)
	idx_x1 = idx_x1_samples3(i);
	semilogy(t, A_adv(idx_x1,:), 'linewidth', 1)
	hold on
end
	
xlabel('Time')
xlim([0 30])
ylabel('Survival')
ylim([10^(-5) 2*10^(1)])
legend(legend_x1_samples3, 'location', 'eastoutside')
title('Survival Curves')
figname = 'SurvivalCurve_t30_samples3.png';
print(gcf, figname, '-dpng')
xlim([0 60])
figname = 'SurvivalCurve_t60_samples3.png';
print(gcf, figname, '-dpng')
xlim([0 90])
figname = 'SurvivalCurve_t90_samples3.png';
print(gcf, figname, '-dpng')
xlim([0 2.5])
ylim([2*10^(-1) 1.1*10^(0)])
figname = 'SurvivalCurve_short_samples3.png';                                                                                                                                                         print(gcf, figname, '-dpng')  
close

figure;
for i = 1:numel(x1_samples4)
	idx_x1 = idx_x1_samples4(i);
	semilogy(t, A_adv(idx_x1,:), 'linewidth', 1)
	hold on
end
	
xlabel('Time')
xlim([0 30])
ylabel('Survival')
ylim([10^(-5) 2*10^(1)])
legend(legend_x1_samples4, 'location', 'eastoutside')
title('Survival Curves')
figname = 'SurvivalCurve_t30_samples4.png';
print(gcf, figname, '-dpng')
xlim([0 60])
figname = 'SurvivalCurve_t60_samples4.png';
print(gcf, figname, '-dpng')
xlim([0 90])
figname = 'SurvivalCurve_t90_samples4.png';
print(gcf, figname, '-dpng')
xlim([0 2.5])
ylim([8*10^(-2) 1.1*10^(0)])
figname = 'SurvivalCurve_short_samples4.png';                                                                                                                                                         print(gcf, figname, '-dpng')  
close
%}

save('Vars_Analysis.mat', 'x1', 's01', 'P1_adv', 'A_adv', 't', 's');
save('Vars_x1s01.mat', 'x1', 's01');







%% FUNCTIONS

%%% PDE w/o drug-induced advection
function[g,f,h] = pdefun_noadv(s,t,P,dPds,b,s_0,sigma,r_fun,k_fun)

	r = r_fun(s);
	k = k_fun(s);

	g = 1;
	f = sigma^2 * dPds + b*(s - s_0)*P;
	h = (r - k) * P;

end

function[g,f,h] = pdefun_adv(s,t,P,dPds,b,s_0,sigma,r_fun,k_fun,v_fun)

	r = r_fun(s);
	k = k_fun(s);
	v = v_fun(s);

	g = 1;
	f = sigma^2 * dPds + b*(s - s_0 - v)*P;
	h = (r - k)*P;
end

%%% Initial condition
function P0 = icfun(s,b,s_0,sigma)

	P0 = 1/sqrt(2*pi*sigma^2 / b) * exp(-b*(s - s_0)^2 / (2*sigma^2));

end

%%% Boundary Condition
function[pl, ql, pr, qr] = bcfun(sl,Pl,sr,Pr,t)

	pl = Pl;
	ql = 0;
	pr = Pr;
	qr = 0;

end 
