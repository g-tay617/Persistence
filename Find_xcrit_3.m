clearvars;

load('Vars_x1s01.mat');

b = 0.09;
sigma = sqrt(b);
tau = 7;        % ctp defined as survival at day 7

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

num_s = numel(s);
num_s01 = numel(s01);
num_t = numel(t);
num_x1 = numel(x1);


%% Green's function method 

vc = 3.5579;

P = zeros(num_x1, num_s, num_t);
parfor i = 1:num_s01
	s_0_i = s01(i);
	for j = 1:num_s
		s_j = s(j);
		for k = 1:num_t
			t_k = t(k);

			P(i,j,k) = sqrt(b/(2*pi*sigma^2)) * exp( -b/(2*sigma^2) * ( (s_j - s_0_i - vc) * exp(b*t_k) + vc )^2 * exp(-2*b*t_k) );
		end
	end
end

gP = zeros(num_x1, num_s, num_t);	% g(s)*P
parfor i = 1:num_s01
	for j = 1:num_s
		s_j = s(j);
		for k = 1:num_t
			gP(i,j,k) = g_fun(s_j) * P(i,j,k);
		end
	end
end

dAdt = zeros(num_x1, num_t);	% With this assumptions/approximations one can show that dA/dt = integral of gP
parfor i = 1:num_s01
	for j = 1:num_t
		dAdt(i,j) = trapz(s,squeeze(gP(i,:,j)));
	end
end

x_c = NaN;
for i = 1:numel(x1)
	
	flag_outer = 0;

	for j = 1:numel(t)-1

		if dAdt(i,j) <= 0 && dAdt(i,j+1) > 0
			flag_outer = 1;
			break;
		end

	end

	if flag_outer == 1
		x_c = x1(i);
		break;
	end

end

figure;
set(gca, 'visible', 'off')
text(0.5, 0.5, ['Using the Greens Function method to approximate yields: x_c = ', num2str(x_c)], 'FontSize', 14, 'HorizontalAlignment', 'center');
saveas(gcf, 'x_critical.png');

disp('Using Green Function Approximation Method:')
disp(['x_c = ' num2str(x_c)])
disp(' ')

%% fully time-dependent PDE

P_real = zeros(num_x1, num_t, num_s);
m = 0;
parfor i = 1:num_x1
	s_0 = s01(i);
	eqn = @(s,t,P,dPds) pdefun_adv(s,t,P,dPds,b,s_0,sigma,r_fun,k_fun,v_fun);
	ic = @(s) icfun(s,b,s_0,sigma);
	bc = @(sl,Pl,sr,Pr,t) bcfun(sl,Pl,sr,Pr,t);
	sol = pdepe(m,eqn,ic,bc,s,t);
	P_real_temp = sol(:,:,1);

	for j = 1:num_s
		for k = 1:num_t
			if P_real_temp(k,j) <= 0
				P_real_temp(k:end,j) = 0;
				break;
			end
		end
	end

	P_real(i,:,:) = P_real_temp;
end

A_real = zeros(length(x1), length(t));
parfor i = 1:numel(x1)
	for j = 1:num_t
		A_real(i,j) = trapz(s, squeeze(P_real(i,j,:)));
	end
end

x1_samples = [0.01:0.01:0.1, 0.2:0.1:0.9];
idx_x1_samples = zeros(size(x1_samples));
legend_x1_samples = cell(size(x1_samples));

for i = 1:numel(x1_samples)
	[~, idx] = min(abs(x1 - x1_samples(i)));
	idx_x1_samples(i) = idx;
	legend_x1_samples{i} = ['$x = $', num2str(x1_samples(i))];
end

figure;
for i = 1:numel(x1_samples)
	idx_x = idx_x1_samples(i);

	semilogy(t, A_real(idx_x,:), 'linewidth', 1)
	hold on
end
hold off	
xlabel('Time', 'fontsize', 14)
ylabel('Survival', 'fontsize', 14)
lgd = legend(legend_x1_samples, 'interpreter', 'latex')
set(lgd, 'location', 'best')
figname = 'SurvivalCurves_fullytimedependent.png';
print(gcf, figname, '-dpng')
close

dAdt_real = zeros(length(s01), length(t)-1);
num_t_2 = num_t - 1
parfor i = 1:num_s01
	for j = 1:num_t_2
		dAdt_real(i,j) = (A_real(i,j+1) - A_real(i,j))/dt;
	end
end

x1_crit_real = NaN;
for i = 3:numel(x1)
	flag_outer = 0;

	for j = 1:numel(t) - 1
		if A_real(i,j) <= A_real(i,j+1) && A_real(i,j+1) - A_real(i,j) > 0.00001
			flag_outer = 1;
			break;
		end
	end

	if flag_outer ==1
		x1_crit_real = x1(i);
		break;
	end

end

figure;
set(gca, 'visible', 'off')
text(0.5, 0.5, ['By solving the fully time-dependent PDE, x_c = ' num2str(x1_crit_real)], 'FontSize', 14, 'HorizontalAlignment', 'center');
saveas(gcf, 'x_crit_real.png')

disp('After solving the fully time-dependent PDE: ')
disp(['x_c = ' num2str(x1_crit_real)])
disp(' ')


%% asmyptotic approximation

dAdt_integrand = zeros(num_x1, num_s);
parfor i = 1:num_x1
	for j = 1:num_s
		s0_x = s01(i);
		s_j = s(j);
		dAdt_integrand(i,j) = ( r_fun(s_j) - k_fun(s_j) ) * sqrt(b/(2*pi*sigma^2)) * exp( -b/(2*sigma^2) * (s_j - s0_x - vc)^2 );
	end
end

dAdt_ss = zeros(1, num_x1);
parfor i = 1:num_x1
	integrand = squeeze(dAdt_integrand(i,:));
	dAdt_ss(i) = trapz(s, integrand);
end

figure;
plot(x1(3:end), dAdt_ss(3:end), 'linewidth', 2)
xlabel('x')
xlim([0 1])
ylabel('Steady-State Value')
title(['Steady-State Values of dA/dt for Various CTP Values'])
figname = 'dAdt_ss.png';
print(gcf, figname, '-dpng')
xlim([0 0.2])
figname = 'dAdt_ss_zoomed.png';
print(gcf, figname, '-dpng')
close

dAdt_ss(1) = dAdt_ss(3);
dAdt_ss(2) = dAdt_ss(3);

[~, idx] = min(abs(dAdt_ss - 0));
xc_ss = x1(idx);

disp(' ')
disp(['xc_ss = ' num2str(xc_ss)])
disp(' ')

figure;
plot(x1, dAdt_ss, 'linewidth', 2)
xlabel('x')
xlim([0 1])
ylabel('Steady-State Value')
title(['Steady-State Values of dA/dt for Various CTP Values'])
figname = 'dAdt_ss_forced.png';
print(gcf, figname, '-dpng')
xlim([0 0.2])
figname = 'dAdt_ss_forced_zoomed.png';
print(gcf, figname, '-dpng')
xlim([0 0.14])
ylim([-0.2 0.1])
figname = 'dAdt_ss_forced_zoomed2.png';
print(gcf, figname, '-dpng')
close

save('FinalData.mat', 'x1', 's01', 'dAdt_ss', 't', 's', 'dAdt', 'dAdt_real')


delete(gcp('nocreate'));

%% FUNCTIONS

%%% Fully time-dependent PDE with drug-induced advection
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
function[pl,ql,pr,qr] = bcfun(sl,Pl,sr,Pr,t)

	pl = Pl;
	ql = 0;
	pr = Pr;
	qr = 0;

end
