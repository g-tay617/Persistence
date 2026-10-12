%%% PARAMETERS
a_tilde = 17^3;
x_bar = 0.067;

Pss_X = @(ctp) airy(a_tilde^(1/3)*(ctp - x_bar));
A_Pss = 1/Pss_X(0);
Pss_X = @(ctp) A_Pss * Pss_X(ctp);

b = 0.09;
sigma = sqrt(b);
tau = 7;

mu = 0.000036;
a = mu * a_tilde;
xi = a*x_bar;
r_x = 0.2;

%%% Grids
ds = 0.01;
s_left = -50;
s_right = 50;
s = s_left:ds:s_right;
s = s.';

dt = 0.1;
t_left = 0;
t_right = 90;
t = t_left:dt:t_right;
t = t.';

s0_left = -10;
s0_right = 10;
s0 = s0_left:ds:s0_right;
s0 = s0.';

%%% Drug concentration 
c = 1;

%%% proliferation
r_0 = 0.08;
s_r = 2.5;
alpha = 1;

Nss = 1;

Rs_unsat = @(s) r_0 ./ (1 + exp(-alpha*(s - s_r)));
Rs = @(s,Nt) Rs_unsat(s) * (1 - Nt/Nss);
Rx = @(x,Nt) xi - a*x + r_x*(1 - Nt/Nss);
c_R = 1e-6;
r_fun = @(x,s,Nt) Rx(x,Nt)./(1 + c/c_R) + Rs(s,Nt)./(1 + c_R/c);

%%% killing
kmax = 1;
c_50 = 0.05;
gamma = 4;
s_k = 0;
k_0 = kmax / (1 + c_50/c);

k_fun = @(s) k_0 ./ (1 + exp(gamma*(s - s_k)));

%%% advection

h = 1;
c_v = 0.05;	% a pretty sensitive parameter. Different cell line might have a different value
beta = 0.5;
vmax = 10;
s_v = 0;
v_0 = vmax*c/(c_v + c);

v_fun = @(s) v_0 ./ (1 + exp(beta*(s - s_v)));

num_s = numel(s);
num_s0 = numel(s0);
num_t = numel(t);

Rs_base = r_0 ./ (1 + exp(-alpha*(s - s_r)));
k_grid = k_0 ./ (1 + exp(gamma*(s - s_k)));
v_grid = v_0 ./ (1 + exp(beta*(s - s_v)));

Nss = 1;	% steady-state/carrying capacity

[~, idxtau] = min(abs(t - tau));

% diffusion operator
uno = ones(num_s, 1);
Lap = spdiags([uno -2*uno uno], [-1 0 1], num_s, num_s)/(ds^2);
Lap(1,1) = -2/(ds^2);
Lap(1,2) = 2/(ds^2);
Lap(end,end) = -2/(ds^2);
Lap(end,end-1) = 2/(ds^2);	% Neumann boundary condition

I = speye(num_s);
AhalfL = I - (1/2)*dt*sigma^2*Lap;
AhalfR = I + (1/2)*dt*sigma^2*Lap;

[Lfac, Ufac, Pperm] = lu(AhalfL);	% LU factorization
solve_half_diff = @(Y) Ufac \ (Lfac \ (Pperm*(AhalfR*Y)));

icfun = @(sgrid,s0) ( 1/sqrt(2*pi*sigma^2 / b) ) * exp( (-b/(2*sigma^2)) * (sgrid - s0).^2 );

x_out = zeros(num_s0, 1);

parpool(32, 'IdleTimeout', Inf);

parfor i = 1:num_s0

	s_0 = s0(i);

	P = icfun(s, s_0);	% initial 
	P = max(P,0);

	N0 = trapz(s, P);

	for n = 1:idxtau-1

		N_now = trapz(s, P);
		g = 1 - N_now/Nss;	% logistic term
		
		P = solve_half_diff(P);	% half-diffusion

		u = -b*(s - s_0 - v_grid);
		P = SL_advect_1D(P, s, dt, u);

		du = deriv_1D(u, ds);	% du/ds
		
		%Rx_x = xi - a*x + r_x*g;	% num_x x 1
		%RxS = repmat(Rx_x(:).', num_s, 1);	% num_s x num_x
		
		%w_Rx = 1/(1 + c/c_R);
		%w_Rs = 1/(1 + c_R/c);

		%r_grid = w_Rx*Rxs + w_Rs*repmat(Rs_base*g, 1, num_x);
		%r_grid = repmat(Rs_base*g, 1, num_x);

		%net = (r_grid - repmat(k_grid, 1, num_x)) - du;	% (r - k) - du
		net = (Rs_base*g - k_grid) - du;
		P = P .* exp(net*dt);	% the reaction step with compressibility correction

		P = solve_half_diff(P);	% half-diffusion again 

		P = max(P,0);	% everything should be >= 0

	end

	x_out(i) = trapz(s, P)/N0;

end

figure;
plot(x_out, s0, 'linewidth', 2.5)
xlabel('$x$', 'interpreter', 'latex', 'fontsize', 16)
ylabel('$s_0$', 'interpreter', 'latex', 'fontsize', 16, 'rotation', 0)
xlim([0 1])
figname = 'plot_s0_x.png';
print(gcf, figname, '-dpng', '-r600')
close

table_x_s0 = table(x_out, s0, 'VariableNames', {'x', 's_0'});
writetable(table_x_s0, 'x_s0.txt');
save('x_s0.mat', 'x_out', 's0');

dx = 0.001;
x1 = 0:dx:1;
x1 = x1.';
s01 = interp1(x_out, s0, x1, 'linear', 'extrap');

x = x1;
s0 = s01;

table_x_s0_interp = table(x, s0, 'VariableNames', {'x', 's_0'});
writetable(table_x_s0_interp, 'x_s0_interp.txt');

save('x_s0_interp.mat', 'x', 's0');
save('Params.mat', 'x', 's0', 'a_tilde', 'x_bar', 'mu', 'a', 'xi', 'r_x', 'b', 'sigma', 'tau', 'r_0', 's_r', 'alpha', 'kmax', 'c_50', 'gamma', 's_k', 'h', 'c_v', 'beta', 'vmax', ...
	's_v', 'c_R');



%% FUNCTIONS

function d = deriv_1D(f, ds)

	d = zeros(size(f));
	d(2:end-1) = (f(3:end) - f(1:end-2))/(2*ds);
	d(1) = (f(2) - f(1))/ds;
	d(end) = (f(end) - f(end-1))/ds;

end

function Yadv = SL_advect_1D(Y, s, dt, u)

	num_s = numel(s);
	s_from = s - u*dt;
	s_from = max(min(s_from, s(end)), s(1));	% clamping

	idx = (s_from - s(1))/(s(2) - s(1)) + 1;
	i0 = floor(idx);
	i0 = max(i0,1);
	i1 = min(i0+1,num_s);
	w1 = idx - i0;
	w0 = 1 - w1;

	Yadv = w0.*Y(i0) + w1.*Y(i1);

end

