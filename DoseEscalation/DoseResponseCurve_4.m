T_viab = 9;	% Kuramochi, Ovasho, COV362
%T_viab = 4;	% A375
%T_viab = 5;	% PC9

load('Params.mat');

x = x(:);
s0 = s0(:);

ds = 0.01;
s_left = -50;
s_right = 50;
s = (s_left:ds:s_right).';

num_s = numel(s);
num_x = numel(x);

dt = 0.1;
t = (0:dt:T_viab).';
num_t = numel(t);

Pss_X_raw = @(ctp) airy(a_tilde^(1/3) * (ctp - x_bar));
normconst = 1/integral(Pss_X_raw, 0, 1);
Pss_X = @(ctp) normconst * Pss_X_raw(ctp);
Px = Pss_X(x);
Px = Px(:);

logc = linspace(-5, 1, 10);
concs = 10.^logc;
viabilities = zeros(size(concs));

Rs_base = r_0 ./ (1 + exp(-alpha*(s - s_r)));

Nss = integral(Pss_X, 0, 1) *10/7;	% assume that the experiment begins at 70% confluence
N_init = Nss;

icfun = @(sgrid, s0x) ( 1/sqrt(2*pi*sigma^2/b) ) * exp( (-b/(2*sigma^2)) * (sgrid - s0x).^2 );

uno = ones(num_s,1);
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

	c_ref = concs(i_c);

	k_0 = kmax/(1 + c_50/c_ref);
	k_grid = k_0 ./ (1 + exp(gamma*(s - s_k)));

	v_0 = vmax*c_ref^h/(c_v + c_ref^h);
	v_grid = v_0 ./ (1 + exp(beta*(s - s_v)));

	%c_R = 5e-5;	% this value can differ depending on the cell line. If you don't specify here, it will take the value from InitialAnalysis_8.m
	w_Rx = 1/(1 + c_ref/c_R);
	w_Rs = 1/(1 + c_R/c_ref);

	P = zeros(num_s, num_x);
	for j = 1:num_x

		P(:,j) = Px(j)*icfun(s,s0(j));

	end

	N0 = trapz(x, trapz(s, P, 1));
	P = (N_init/N0) * P;

	for it = 2:num_t

		N_now = trapz(x, trapz(s, P, 1));
		g = 1 - N_now/Nss;

		vS = repmat(v_grid, 1, num_x);
		u = -b*(S - S0 - vS);

		P = SL_advect_2D(P,s,dt,u);

		Rx_x = xi - a*x + r_x;
		RxS = repmat(Rx_x(:).', num_s, 1);
		r_grid = w_Rx*RxS + w_Rs*repmat(Rs_base, 1, num_x);

		du = deriv_2D(u,ds);
		net = (r_grid*g - repmat(k_grid,1,num_x)) - du;

		P = P.*exp(net*dt);

		P = solve_half_diff(P);

		P = max(P,0);

	end

	N_end = trapz(x, trapz(s, P, 1));
	viabilities(i_c) = N_end/N_init;

end

figure;
semilogx(concs, viabilities, 'linewidth', 2)
xlabel('Concentration')
ylabel('Viability')
title(['Viabilities at t = ' num2str(T_viab) ' days'])
figname = ['Viabilities_' num2str(T_viab) 'days.png'];
print(gcf, figname, '-dpng')
close

drCurve_fit_2;

save('Viabilities.mat', 'viabilities', 'concs', 'T_viab', 'IC50', 'nbot');


%% FUNCTIONS

function du = deriv_2D(u,ds)

	du = zeros(size(u));
	du(2:end-1,:) = (u(3:end,:) - u(1:end-2,:))/(2*ds);
	du(1,:) = (u(2,:) - u(1,:))/ds;
	du(end,:) = (u(end,:) - u(end-1,:))/ds;

end

function Yadv = SL_advect_2D(Y, s, dt, u)

	[num_s, num_x] = size(Y);
	S = repmat(s, 1, num_x);
	s_from = S - u*dt;
	s_from = max(min(s_from, s(end)), s(1));	% clamping

	idx = (s_from - s(1))/(s(2) - s(1)) + 1;
	i0 = floor(idx);
	i0 = max(i0,1);
	i1 = min(i0+1, num_s);
	w1 = idx - i0;
	w0 = 1 - w1;

	[~, J] = ndgrid(1:num_s, 1:num_x);
	lin = @(I,J) sub2ind([num_s, num_x], I, J);

	Yadv = w0.*Y(lin(i0,J)) + w1.*Y(lin(i1,J));

end
