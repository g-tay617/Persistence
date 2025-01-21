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

%%% Drug concentrations
logc = -5:0.01:1;
c = 10.^logc;
c = c(1:1*40);

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
k_0 = @(c) kmax ./ (1 + c_50./c);

k_fun = @(s,c) k_0(c) ./ (1 + exp(gamma*(s - s_k)));

%%% advection 
      
c_v = 0.05;
beta = 0.5;
vmax = 10;
s_v = 0;
v_0 = @(c) vmax*c./(c_v + c);
		       
v_fun = @(s,c) v_0(c) ./ (1 + exp(beta*(s - s_v)));
			  
					       
g_fun = @(s,c) r_fun(s) - k_fun(s,c);

s_zeroNet = zeros(size(c));
v_zeroNet = zeros(size(c));
for i = 1:numel(c)
    conc = c(i);
    diff_rk = @(s) g_fun(s,conc);
    s_zero = fzero(diff_rk, 1);
    s_zeroNet(i) = s_zero;

    %P_distr = @(s) 1/sqrt(2*pi*sigma^2/b) * exp( -b/(2*sigma^2) * (s - s_zero).^2);
    %P_distr = @(s) 1/90 * 1/sqrt(2*pi*sigma^2/b) * exp( -b/(2*sigma^2) * (s - s_zero).^2);
    %P_distr = @(s) P_distr(s) / trapz(s, P_distr(s));
    %v_zeroNet(i) = trapz(s, P_distr(s).*v_fun(s,conc));
    v_zeroNet(i) = v_fun(s_zero,conc);
    %v_zeroNet(i) = 3.85;
end

figure;
semilogx(c, s_zeroNet, 'linewidth', 2)
xlabel('$c$', 'interpreter', 'latex', 'fontsize', 16)
ylabel('Value of $s$ at Zero', 'interpreter', 'latex', 'fontsize', 14)
title('$r - k = 0$', 'interpreter', 'latex', 'fontsize', 20)
figname = 's_zeroNet.png';
print(gcf, figname, '-dpng')
close

figure;
semilogx(c, v_zeroNet, 'linewidth', 2)
xlabel('$c$', 'interpreter', 'latex', 'fontsize', 16)
ylabel('$v$ when $r - k = 0$', 'interpreter', 'latex', 'fontsize', 16)
%title('$v$ when $r - k = 0$', 'interpreter', 'latex', 'fontsize', 20)
figname = 'v_zeroNet.png';
print(gcf, figname, '-dpng')
close

xc = zeros(size(c));
dAdt_ss_values = zeros(size(c));
for i = 1:numel(c)
    conc = c(i);
    vc = v_zeroNet(i);

    dAdt_integrand = zeros(numel(x1), numel(s));
    for j = 1:numel(x1)
        s0_x = s01(j);
        for k = 1:numel(s)
            dAdt_integrand(j,k) = ( r_fun(s(k)) - k_fun(s(k), conc) ) * sqrt(b/(2*pi*sigma^2)) ...
                * exp( -b/(2*sigma^2) * (s(k) - s0_x - vc)^2 );
        end
    end

    dAdt_ss = zeros(1, numel(x1));
    for j = 1:numel(x1)
        integrand = squeeze(dAdt_integrand(j,:));
        dAdt_ss(j) = trapz(s, integrand);
    end

    dAdt_ss = dAdt_ss(3:end);

    [val, idx] = min(abs(dAdt_ss));
    xc(i) = x1(idx+2);
    dAdt_ss_values(i) = val;

end

figure;
semilogx(c, xc, 'linewidth', 2)
xlabel('$c$', 'interpreter', 'latex', 'fontsize', 16)
ylabel('$x_c$', 'interpreter', 'latex', 'fontsize', 16)
title('Our Approximation Method')
figname = 'xc_ourapproxmethod.png';
print(gcf, figname, '-dpng')
close

%% Now, we find the actual value using fully time-dependent pde 
m = 0;
parpool(128);

xc_real = zeros(size(c));

for i = 1:numel(c)

	conc = c(i);

	v_fun_conc = @(s) v_fun(s,conc);
	k_fun_conc = @(s) k_fun(s,conc);	

	P_real = zeros(numel(x1), numel(t), numel(s));
	parfor j = 1:numel(x1)
		s_0 = s01(j);
		eqn = @(s,t,P,dPds) pdefun_adv(s,t,P,dPds,b,s_0,sigma,r_fun,k_fun_conc,v_fun_conc);
		ic = @(s) icfun(s,b,s_0,sigma);
		bc = @(sl,Pl,sr,Pr,t) bcfun(sl,Pl,sr,Pr,t);
		sol = pdepe(m,eqn,ic,bc,s,t);
		P_real_temp = sol(:,:,1);

		for k = 1:numel(s)
			for l = 1:numel(t)
			
				if P_real_temp(l,k) <= 0
					P_real_temp(l:end,k) = 0;
					break;
				end

			end
		end

		P_real(j,:,:) = P_real_temp;

	end

	A_real = zeros(length(x1), length(t));
	for j = 1:numel(x1)
		for k = 1:numel(t)
			A_real(j,k) = trapz(s, squeeze(P_real(j,k,:)));
		end
	end

	dAdt_real = zeros(length(s01), length(t)-1);
	for j = 1:numel(s01)
		for k = 1:numel(t)-1
			dAdt_real(j,k) = (A_real(j,k+1) - A_real(j,k))/dt;
		end
	end

	xc_real_i = NaN;
	for j = 3:numel(x1)
		flag_outer = 0;

		for k = 1:numel(t) - 1

			if A_real(j,k) <= A_real(j,k+1) && A_real(j,k+1) - A_real(j,k) > 0.00001
				flag_outer = 1;
				break;
			end

		end

		if flag_outer == 1
			xc_real_i = x1(j);
			break;
		end

	end

	xc_real(i) = xc_real_i;

end

figure;
semilogx(c, xc, 'linewidth', 2)
hold on
semilogx(c, xc_real, 'linewidth', 2)
hold off
xlabel('$c$', 'interpreter', 'latex', 'fontsize', 16)
ylabel('$x_c$', 'interpreter', 'latex', 'fontsize', 16, 'rotation', 0)
legend({'Predicted', 'PDE Solution'}, 'location', 'best')
figname = 'x_critical.png';
print(gcf, figname, '-dpng')
close

save('FinalData.mat', 'c', 'xc', 'xc_real', 's_zeroNet', 'v_zeroNet')
		
		




%% FUNCTIONS

function[g,f,h] = pdefun_adv(s,t,P,dPds,b,s_0,sigma,r_fun,k_fun_conc,v_fun_conc)

	r = r_fun(s);
	k = k_fun_conc(s);
	v = v_fun_conc(s);

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




