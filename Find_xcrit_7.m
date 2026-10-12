load('Params.mat');	% run InitialAnalysis_8.m first!

load('ConstantDoseResults.mat');	% also run ConstantDose_4.m first (make sure the concentration of interest is set accordingly there)

num_x = numel(x);
num_s = numel(s);
num_t = numel(t);
dt = t(2) - t(1);

Rs_base = r_0 ./ (1 + exp(-alpha*(s - s_r)));
k_grid = k_0 ./ (1 + exp(gamma*(s - s_k)));
v_grid = v_0 ./ (1 + exp(beta*(s - s_v)));

x_interest = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75];
idx_x_interest = zeros(size(x_interest));
legend_x_interest = cell(size(x_interest));

for i = 1:numel(x_interest)

	[~, idx] = min(abs(x - x_interest(i)));
	idx_x_interest(i) = idx;
	legend_x_interest{i} = ['$x = ' num2str(x_interest(i)) '$'];

end

g_t = 1 - N_t./Nss;	% logistic factor
 
dAdt = zeros(num_x, num_t);


%parpool(128, 'IdleTimeout', Inf);

parfor i = 1:num_x

	for j = 1:num_t

		P_s = squeeze(P_xts(i,j,:));
		integrand = (Rs_base.*g_t(j) - k_grid) .* P_s;
		dAdt(i,j) = trapz(s, integrand);

	end

end

x_c = NaN;	% find the critical x_c -- the lowest x with steady-state dAdt > 0 (by steady-state within this simulation timescale). 

for i = 1:num_x
	
	if any(dAdt(i,:) > 0)
		x_c = x(i);
		break;
	end

end

disp(['By fully solving the PDE, x_c = ' num2str(x_c)]);

% Using the Green's function

k_0 = kmax/(1 + c_50/c_ref);
v_0 = vmax*c_ref^h/(c_v + c_ref^h);

Rs_fun = @(s) r_0 ./ (1 + exp(-alpha*(s - s_r)));
k_fun = @(s) k_0 ./ (1 + exp(gamma*(s - s_k)));
v_fun = @(s) v_0 ./ (1 + exp(beta*(s - s_v)));

s_zero = fzero(@(ss) Rs_fun(ss) - k_fun(ss), 1);
v_c = v_fun(s_zero);	% we choose some constant v_c that represent v when the net change ~ 0

P_G = zeros(num_x, num_t, num_s);

parfor i = 1:num_x

	s_0_i = s0(i);

	for j = 1:num_t

		t_j = t(j);

		for k = 1:num_s

			s_val = s(k);

			P_G(i,j,k) = sqrt(b/(2*pi*sigma^2)) * exp( -b/(2*sigma^2) * ( (s_val - s_0_i - v_c) * exp(b*t_j) + v_c )^2 * exp(-2*b*t_j) );

		end

	end

end

dAdt_G = zeros(num_x, num_t);

parfor i = 1:num_x

	for j = 1:num_t

		P_s = squeeze(P_G(i,j,:));
		integrand = (Rs_base.*g_t(j) - k_grid) .* P_s;
		dAdt_G(i,j) = trapz(s, integrand);

	end

end


x_c_G = NaN;
for i = 1:num_x

	%{
    flag_outer = 0;

    for j = 1:num_t-1

        if dAdt_G(i,j) <= 0 && dAdt_G(i,j+1) > 0
            flag_outer = 1;
            break;
        end

    end

    if flag_outer == 1
        x_c_G = x(i);
        break;
    end
	%}

	if any(dAdt_G(i,:) > 0)
		x_c_G = x(i);
		break;
	end

end

disp(['With our semi-analytical approximation method, x_c_G = ' num2str(x_c_G)]);

% Sanity check with actual A_xt values. 
figure;
for i = 1:numel(x_interest)

	i_x = idx_x_interest(i);

	semilogy(t, A_xt(i_x,:), 'linewidth', 1.5)
	hold on

end
hold off
xlabel('Time', 'interpreter', 'latex', 'fontsize', 16)
ylabel('Survival Fraction', 'interpreter', 'latex', 'fontsize', 16)
lgd = legend(legend_x_interest, 'interpreter', 'latex')
set(lgd, 'location', 'best')
figname = 'A_xt_interest.png';
print(gcf, figname, '-dpng')
close

dAdt_real = zeros(num_x, num_t);

for i = 1:num_x

	dAdt_real(i,1:num_t-1) = ( A_xt(i,2:num_t) - A_xt(i,1:num_t-1) )/dt;
	dAdt_real(i,end) = dAdt_real(i,end-1);

end

figure;
for i = 1:numel(x_interest)

	i_x = idx_x_interest(i);

	plot(t, dAdt_real(i_x,:), 'linewidth', 1.5)
	hold on

end
hold off
xlabel('Time', 'interpreter', 'latex', 'fontsize', 16)
ylabel('Survival Fraction', 'interpreter', 'latex', 'fontsize', 16)
lgd = legend(legend_x_interest, 'interpreter', 'latex')
set(lgd, 'location', 'best')
figname = 'dAdt_interest.png';
print(gcf, figname, '-dpng')
close

x_c_real = NaN;
for i = 1:num_x

	if any(dAdt_real(i,:) > 0)
		x_c_real = x(i);
		break;
	end

end

disp(['By manually using A_xt, x_c = ' num2str(x_c_real)]);

% asymptotic approximation

dAdt_ss = zeros(size(x));
g_ss = g_t(end);

parfor i = 1:num_x

	s_0 = s0(i);
	integrand = (Rs_base*g_ss - k_grid) .* sqrt(b/(2*pi*sigma^2)) .* exp(-b/(2*sigma^2) * (s - s_0 - v_c).^2);
	integrand = squeeze(integrand);
	dAdt_ss(i) = trapz(s, integrand);

end

idx = find(dAdt_ss > 0, 1, 'first');
if isempty(idx)
	x_c_ss = NaN;
else
	if idx == 1
		x_c_ss = x(1);
	else
		x_c_ss = interp1( dAdt_ss(idx-1:idx), x(idx-1:idx), 0, 'linear');
	end
end

disp(['Asymptotic approximation yields x_c = ' num2str(x_c_ss)])

figure;
plot(x(2:end), dAdt_ss(2:end), 'linewidth', 2.5)	% remove x = 0 (unstable)
xlabel('$x$', 'interpreter', 'latex', 'fontsize', 16)
ylabel('Steady-State $\frac{dA}{dt}$', 'interpreter', 'latex', 'fontsize', 16)
hold on
yline(0, '--', 'linewidth', 1.5)
xlim([0 1])
figname = 'dAdt_ss.png';
print(gcf, figname, '-dpng')
close

save('Find_xcrit_results.mat', 'dAdt', 'dAdt_G', 'dAdt_real', 'dAdt_ss');
save('xcrits.mat', 'x_c', 'x_c_G', 'x_c_real', 'x_c_ss');







