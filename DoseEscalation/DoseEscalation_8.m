%% Data from the Yanai paper

% uncomment one of these cells 

% concs_olaparib_Kuramochi = [1, 2.5, 5, 10, 20, 40, 80, 160, 320];
% concs_data = concs_olaparib_Kuramochi;
% cycles_Kuramochi = 1:numel(concs_olaparib_Kuramochi);
% cycles = cycles_Kuramochi;
% cumdays_Kuramochi = [6, 16, 39, 69, 119, 139, 179, 224, 271, 311];
% T_conf_Kuramochi = diff(cumdays_Kuramochi);
% T_conf_data = T_conf_Kuramochi;
% viab_1 = 0.95;
% 
% concs_olaparib_Ovasho = [1, 2.5, 5, 10, 20, 40];
% concs_data = concs_olaparib_Ovasho
% cycles_Ovasho = 1:numel(concs_olaparib_Ovasho);
% cycles = cycles_Ovasho;
% cumdays_Ovasho = [7, 20, 38, 59, 83, 111, 125];
% T_conf_Ovasho = diff(cumdays_Ovasho);
% T_conf_data = T_conf_Ovasho;
% viab_1 = 0.9;
% 
concs_olaparib_COV362 = [1, 2.5, 5, 10, 20, 40];
concs_data = concs_olaparib_COV362
cycles_COV362 = 1:numel(concs_olaparib_COV362);
cycles = cycles_COV362;
cumdays_COV362 = [6, 34, 56, 80, 120, 146, 162];
T_conf_COV362 = diff(cumdays_COV362);
T_conf_data = T_conf_COV362;
viab_1 = 0.65;
% 
% concs_dabrafenib_A375 = [0.03, 0.06, 0.125, 0.25, 0.5, 1, 2, 4];
% concs_data = concs_dabrafenib_A375;
% cycles_A375 = 1:numel(concs_dabrafenib_A375);
% cycles = cycles_A375;
% cumdays_A375 = [5, 15, 22, 28, 34, 41, 48, 53, 58];
% T_conf_A375 = diff(cumdays_A375);
% T_conf_data = T_conf_A375;
% viab_1 = 0.35;
% 
% concs_osimertinib_PC9 = [0.008, 0.016, 0.032, 0.075, 0.15, 0.3, 0.6, 1.2];
% concs_data = concs_osimertinib_PC9;
% cycles_PC9 = 1:numel(concs_osimertinib_PC9);
% cycles = cycles_PC9;
% cumdays_PC9 = [5, 12, 21, 30, 37, 45, 51, 58, 68];
% T_conf_PC9 = diff(cumdays_PC9);
% T_conf_data = T_conf_PC9;
% viab_1 = 0.4;


DoseResponseCurve_4;

%load('Params.mat');	% can comment it out if already loaded in DoseResponseCurve_4.m
%load('Viabilities.mat');

dr = @(c) nbot + 1./(1 + c./IC50);

% choose first dose by solving dr(c) = viab1. Or you can choose your own.
%viab1 = 0.95;
viab1 = viab_1;
firstdose = fzero(@(c) dr(c) - viab1, [1e-5, 10]);
%firstdose = 1e-3;

x = x(:);
s0 = s0(:);

ds = 0.01;
s_left = -50;
s_right = 50;
s = (s_left:ds:s_right).';

num_s = numel(s);
num_x = numel(x);

dt = 0.1;

Pss_X_raw = @(ctp) airy(a_tilde^(1/3) * (ctp - x_bar));
normconst = 1/integral(Pss_X_raw, 0, 1);
Pss_X = @(ctp) normconst * Pss_X_raw(ctp);
Px = Pss_X(x);
Px = Px(:);

Rs_base = r_0 ./ (1 + exp(-alpha*(s - s_r)));

% this can be changed depending on the experimental protocol - can use the uncommented values at the top
% you can use the below for general simulations
num_cycles = numel(cycles);
%concs = zeros(1, num_cycles);
%concs(1) = firstdose;
%concs(2) = firstdose*2.5;
%for i = 3:num_cycles
%
%	concs(i) = concs(2)*2^(i-2);
%
%end
multipliers_concs = concs_data/concs_data(1);
concs = firstdose * multipliers_concs;

legend_concs = cell(size(concs));
for i = 1:numel(concs)

	legend_concs{i} = ['c = ' num2str(concs(i))];

end

icfun = @(sgrid,s0x) 1/sqrt(2*pi*sigma^2 / b) * exp( (-b/(2*sigma^2)) * (sgrid - s0x).^2 );

P = zeros(num_s, num_x);
for j = 1:num_x

	P(:,j) = Px(j)*icfun(s,s0(j));

end

N_init = trapz(x, trapz(s,P,1));
Px_init = trapz(s,P,1).';
Ps_init = trapz(x,P,2);
Px_init = Px_init / N_init;
Ps_init = Ps_init / N_init;

Px_cycles = zeros(num_x, num_cycles);
Ps_cycles = zeros(num_s, num_cycles);
N_cycles = zeros(1, num_cycles);

% confluency threshold 
T_confluency = zeros(size(concs));
N_confluent = zeros(size(concs));
i_max = 9;
N_max = 10;
N_min = 5;
for i = 1:i_max

	N_confluent(i) = N_min + (N_max - N_min)/(i_max - 1) * (i-1);	% based on my convo with the first author of the Yanai paper

end
N_confluent(i_max:end) = N_confluent(i_max);
N_eqn = 1;	% rescale P to this at the end of each cycle
conf_thresh = 0.7;	% we assume that >70% of Nss means confluent

uno = ones(num_s, 1);
Lap = spdiags([uno -2*uno uno], [-1 0 1], num_s, num_s)/(ds^2);
Lap(1,1) = -2/(ds^2);
Lap(1,2) = 2/(ds^2);
Lap(end,end) = -2/(ds^2);
Lap(end,end-1) = 2/(ds^2);

I = speye(num_s);
AhalfL = I - 0.5*dt*sigma^2*Lap;
AhalfR = I + 0.5*dt*sigma^2*Lap;

[Lfac, Ufac, Pperm] = lu(AhalfL);
solve_half_diff = @(Y) Ufac \ (Lfac \ (Pperm*(AhalfR*Y)));

S = repmat(s, 1, num_x);
S0 = repmat(s0(:).', num_s, 1);

for i_c = 1:numel(concs)

	conc = concs(i_c);

	k_0 = kmax/(1 + c_50/conc);
	k_grid = k_0./(1 + exp(gamma*(s - s_k)));

	v_0 = vmax*conc^h/(c_v + conc^h);
	v_grid = v_0./(1 + exp(beta*(s - s_v)));

	w_Rx = 1/(1 + conc/c_R);
	w_Rs = 1/(1 + c_R/conc);

	Nss = N_confluent(i_c)/conf_thresh;

	N_now = trapz(x, trapz(s, P, 1));
	P = (N_eqn/N_now)*P;	% start with N_eqn
	N_now = N_eqn;
	Confluent = false;
	t_abs = 0;

	%disp(['Beginning Cycle #' num2str(i_c) ' with c = ' num2str(conc)' ', Nss = ' num2str(Nss)])
	disp(sprintf('Beginning Cycle # %d with c = %.6g, Nss = %.6g', i_c, conc, Nss));

	step = 0;

	while ~Confluent

		step = step + 1;

		g = 1 - N_now/Nss;

		P = solve_half_diff(P);

		vS = repmat(v_grid, 1, num_x);
		u = -b*(S - S0 - vS);
		P = SL_advect_2D(P, s, dt, u);

		Rx_x = xi - a*x + r_x*g;
		RxS = repmat(Rx_x(:).', num_s, 1);
		r_grid = w_Rx*RxS + w_Rs*repmat(Rs_base*g, 1, num_x);

		du = deriv_2D(u,ds);
		net = (r_grid - repmat(k_grid, 1, num_x)) - du;

		P = P.*exp(net*dt);

		P = solve_half_diff(P);

		P = max(P,0);

		N_now = trapz(x, trapz(s, P, 1));
		t_abs = t_abs + dt;

		if N_now >= N_confluent(i_c)
			Confluent = true;
			T_confluency(i_c) = t_abs;
			%disp(['Confluency for Cycle #' num2str(i_c) ' reached at t = ' num2str(t_abs) ' and N = ' num2str(N_now)])
			disp(sprintf('Confluency for Cycle #%d reached at t = %.6g and N = %.6g', i_c, t_abs, N_now));
			break;
		end

		if mod(step, round(50/dt)) == 0
			%disp(fprintf('On cycle #%d, t = %.1f elapsed, N = $.4g', i_c, t_abs, N_now))			
			disp(['On Cycle #' num2str(i_c) ', t = ' num2str(t_abs) ', N_now = ' num2str(N_now)])
		end

	end

	Nt_conf = N_now;
	P_end = P;

	Px_end = trapz(s, P_end, 1);
	Ps_end = trapz(x, P_end, 2);

	Px_end = Px_end(:)/Nt_conf;
	Ps_end = Ps_end(:)/Nt_conf;

	Px_cycles(:,i_c) = Px_end;
	Ps_cycles(:,i_c) = Ps_end;
	N_cycles(i_c) = Nt_conf;

end

figure;
semilogx(concs, T_confluency, '-o', 'linewidth', 2.5)
xlabel('Doses', 'interpreter', 'latex', 'fontsize', 18)
ylabel('Time to Confluency', 'interpreter', 'latex', 'fontsize', 18)
figname = 'T_confluency.png';
print(gcf, figname, '-dpng', '-r600')
close

legend_cycles = cell(1,num_cycles+1);
legend_cycles{1} = 'Naive';
for i = 1:num_cycles
	
	legend_cycles{i+1} = ['Cycle #' num2str(cycles(i))];

end

figure;
hold on
plot(x, Px_init, 'k--', 'linewidth', 1.5)
for i_c = 1:num_cycles
	plot(x, Px_cycles(:,i_c), 'linewidth', 1.5)
end
hold off
xlabel('$x$', 'interpreter', 'latex', 'fontsize', 18)
xlim([0,1])
ylabel('$P_x(x)$', 'interpreter', 'latex', 'fontsize', 18)
legend(legend_cycles, 'location', 'best')
figname = 'Px_distr.png';
print(gcf, figname, '-dpng', '-r600')
close

figure;
hold on
plot(s, Ps_init, 'k--', 'linewidth', 1.5)
for i_c = 1:num_cycles
	plot(s, Ps_cycles(:,i_c), 'linewidth', 1.5)
end
hold off
xlabel('$s$', 'interpreter', 'latex', 'fontsize', 18)
xlim([-10,10])
ylabel('$P_s(s)$', 'interpreter', 'latex', 'fontsize', 18)
legend(legend_cycles, 'location', 'best')
figname = 'Ps_distr.png';
print(gcf, figname, '-dpng', '-r600')
close



save('DoseEscalation_8_results.mat', 'T_confluency', 'concs', 'num_cycles', 'T_conf_data', 'x', 's', 'Px_init', 'Px_cycles', 'Ps_init', 'Ps_cycles');



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
        s_from = max(min(s_from, s(end)), s(1));        % clamp

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
