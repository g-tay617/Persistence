%%%%% After running InitialAnalysis_8.m!!!!%%%%

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
t = (0:dt:100).';

num_t = numel(t);

c_ref = 1;	% drug dose

Pss_X_raw = @(ctp) airy(a_tilde^(1/3) * (ctp - x_bar));
normconst = 1/integral(Pss_X_raw, 0, 1);
Pss_X = @(ctp) normconst * Pss_X_raw(ctp);
Px = Pss_X(x);
Px = Px(:);

Rs_base = r_0 ./ (1 + exp(-alpha*(s - s_r)));

k_0 = kmax/(1 + c_50/c_ref);
k_grid = k_0 ./ (1 + exp(gamma*(s - s_k)));

v_0 = vmax*c_ref^h/(c_v + c_ref^h);
v_grid = v_0 ./ (1 + exp(beta*(s - s_v)));

Nss = 1;
N_init = Nss;

icfun = @(sgrid,s0x) 1/sqrt(2*pi*sigma^2 / b) * exp( (-b/(2*sigma^2)) * (sgrid - s0x).^2 );

P = zeros(num_s, num_x);

for j = 1:num_x

	P(:,j) = Px(j)*icfun(s,s0(j));

end

N0 = trapz(x, trapz(s, P, 1));
P = (N_init/N0) * P;	% normalize to N_init

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

A_xt = zeros(num_x, num_t);	% clonal survival
N_t = zeros(num_t, 1);	% total population 

Px_mass0 = trapz(s, P, 1);
A_xt(:,1) = Px_mass0(:) ./ max(Px_mass0(:), eps);	% should be all 1
N_t(1) = trapz(x, Px_mass0);

P_xts = zeros(num_x, num_t, num_s);
P_xts(:,1,:) = permute(P, [2 3 1]);

S = repmat(s, 1, num_x);
S0 = repmat(s0(:).', num_s, 1);

% initial distribution plot
figure;
imagesc(x, s, P)
set(gca,'YDir', 'normal')
axis tight
colormap(jet)
cb = colorbar;
cb.FontSize = 12;
xlabel('$x$', 'interpreter', 'latex')
ylabel('$s$', 'interpreter', 'latex')
xlim([0 0.45])
ylim([-7 7])
ax = gca;
ax.FontSize = 14;
ax.XLabel.FontSize = 18;
ax.YLabel.FontSize = 18;
figname = 'P_xs_t1.png';
print(gcf, figname, '-dpng', '-r600')
close

plot_every = 10;	% choose frequency

for it = 2:num_t

	N_now = trapz(x, trapz(s, P, 1));
	g = 1 - N_now/Nss;

	P = solve_half_diff(P);

	vS = repmat(v_grid, 1, num_x);
	u = -b*(S - S0 - vS);

	P = SL_advect_2D(P, s, dt, u);

	du = deriv_2D(u,ds);

	w_Rx = 1/(1 + c_ref/c_R);
	w_Rs = 1/(1 + c_R/c_ref);

	Rx_x = xi - a*x + r_x*g;
	RxS = repmat(Rx_x(:).', num_s, 1);
	r_grid = w_Rx*RxS + w_Rs*repmat(Rs_base*g, 1, num_x);

	net = (r_grid - repmat(k_grid,1,num_x)) - du;

	P = P.*exp(net*dt);

	P = solve_half_diff(P);

	P = max(P,0);

	Px_mass = trapz(s, P, 1);
	A_xt(:,it) = (Px_mass(:) ./ Px_mass0(:));
	N_t(it) = trapz(x, Px_mass);

	P_xts(:,it,:) = permute(P, [2 3 1]);

	% if it > 1 && mod(it-1, plot_every) == 0     % comment this part out if you don't want a bunch of heat maps
	% 	figure;
	% 	imagesc(x,s,P)
	% 	set(gca, 'YDir', 'normal')
	% 	axis tight
	% 	colormap(jet)
	% 	cb = colorbar;
	% 	cb.FontSize = 12;
	% 	xlabel('$x$', 'interpreter', 'latex')
	% 	ylabel('$s$', 'interpreter', 'latex')
	% 	xlim([0 0.45])
	% 	ylim([-7 7])
	% 	ax = gca;
	% 	ax.FontSize = 14;
	% 	ax.XLabel.FontSize = 18;
	% 	ax.YLabel.FontSize = 18;
	% 	figname = ['P_xs_t' num2str(it) '.png'];
	% 	print(gcf, figname, '-dpng')
	% end

end

x_interest = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75];   % choice of x-clones
idx_x_interest = zeros(size(x_interest));
legend_x_interest = cell(size(x_interest));

for i = 1:numel(x_interest)

	[~, idx] = min(abs(x - x_interest(i)));
	idx_x_interest(i) = idx;
	legend_x_interest{i} = ['$x = $' num2str(x_interest(i))];

end

figure;
for i = 1:numel(x_interest)

	idx_x = idx_x_interest(i);
	semilogy(t, A_xt(idx_x,:), 'linewidth', 1.5)
	hold on

end
hold off
xlabel('Time', 'interpreter', 'latex', 'fontsize', 16)
ylabel('Survival Fraction', 'interpreter', 'latex', 'fontsize', 16)
lgd = legend(legend_x_interest, 'interpreter', 'latex');
set(lgd, 'location', 'northwest')
lgd.FontSize = 13;
figname = 'survivalcurves_xclones.png';
print(gcf, figname, '-dpng')
close

figure;
semilogy(t, N_t, 'linewidth', 2.5)
xlabel('Time', 'interpreter', 'latex', 'fontsize', 16)
ylabel('Survival Fraction', 'interpreter', 'latex', 'fontsize', 16)
figname = 'totalpopulation.png';
print(gcf, figname, '-dpng')
close

Px_init = trapz(s, squeeze(P_xts(:,1,:)), 2);
Px_init = Px_init(:);
Px_init = Px_init/trapz(x, Px_init);

Px_end = trapz(s, squeeze(P_xts(:,end,:)), 2);
Px_end = Px_end(:);
Px_end = Px_end/trapz(x, Px_end);

figure;
hold on
plot(x, Px_init, 'linewidth', 2, 'color', 'blue')
plot(x, Px_end, 'linewidth', 2, 'color', 'red')
hold off
xlabel('$x$', 'interpreter', 'latex', 'fontsize', 16)
ylabel('Normalized Distribution', 'interpreter', 'latex', 'fontsize', 16)
legend({'Initial', 'Final'})
figname = 'Px_end.png';
print(gcf, figname, '-dpng')
close

Ps_init = trapz(x, squeeze(P_xts(:,1,:)), 1);
Ps_init = Ps_init(:);
Ps_init = Ps_init/trapz(s, Ps_init);

Ps_end = trapz(x, squeeze(P_xts(:,end,:)), 1);
Ps_end = Ps_end(:);
Ps_end = Ps_end/trapz(s, Ps_end);

figure;
hold on
plot(s, Ps_init, 'linewidth', 2, 'color', 'blue')
plot(s, Ps_end, 'linewidth', 2, 'color', 'red')
hold off
xlabel('$s$', 'interpreter', 'latex', 'fontsize', 16)
xlim([-10 10])
ylabel('Normalized Distribution', 'interpreter', 'latex', 'fontsize', 16)
legend({'Initial', 'Final'})
figname = 'Ps_end.png';
print(gcf, figname, '-dpng')
close

%save('P_xts.mat', 'P_xts', '-v7.3');
save('Survs_and_Distributions.mat', 'A_xt', 'N_t', 'Nss', 'Px_init', 'Px_end', 'Ps_init', 'Ps_end', '-v7.3');
save('ConstantDoseResults.mat', '-v7.3');


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
