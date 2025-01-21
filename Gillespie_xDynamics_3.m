a_tilde = 17^3;
mu = 0.00025;
x_bar = 0.067;

Pss_X = @(ctp) airy(a_tilde^(1/3) * (ctp - x_bar));
Pss_X = @(ctp) Pss_X(ctp)/Pss_X(x_bar);

dx = 0.001;
x = 0:dx:1;

[~, idx_x_bar] = min(abs(x - x_bar));

rate_trans = mu/(2*dx^2);
alpha_kill = a_tilde * mu;

N = 2500;
x_init = 0.1;	% all cells start with x = 0.1

xTracker = cell(N,1);	% for recording x positions for each cell
tTracker = cell(N,1);	% for recording time for each cell

t_max = 10000;

parfor i = 1:N

	x_current = x_init;
	t_current = 0;

	x_i_Tracker = x_current;
	t_i_Tracker = t_current;

	stream = RandStream('mrg32k3a', 'Seed', i);

	while t_current < t_max

		rate_xup = rate_trans;
		rate_xdown = rate_trans;
		rate_death = alpha_kill * (x_current - x_bar);

		if rate_death < 0
			rate_death = 0;
		end

		if x_current <= 0
			rate_xdown = 0;
		end
		if x_current >= 1
			rate_xup = 0;
		end

		propensities_total = rate_xup + rate_xdown + rate_death;
	
		r1 = rand(stream);
		dt = -log(r1) / propensities_total;
		t_current = t_current + dt;

		if t_current > t_max
			break;
		end

		r2 = rand(stream) * propensities_total;
		death = false;
		
		if r2 < rate_xup
			x_current = x_current + dx;
		elseif r2 < rate_xup + rate_xdown
			x_current = x_current - dx;
		else
			death = true;

			x_i_Tracker(end+1) = NaN;
			t_i_Tracker(end+1) = NaN;

			x_current = rand(stream);
			[~, idx_x_rand] = min(abs(x - x_current));
			x_current = x(idx_x_rand);
			t_current = t_current;
		end

		x_current = max([min([x_current, 1]), 0]);
		
		x_i_Tracker(end+1) = x_current;
		t_i_Tracker(end+1) = t_current;

		%if death
		%	x_i_Tracker(end+1) = NaN;
		%	t_i_Tracker(end+1) = NaN;
		%end
	end

	xTracker{i} = x_i_Tracker;
	tTracker{i} = t_i_Tracker;

end

% Plot trajectories of all of them
figure;
for i = 1:N
	plot(tTracker{i}, xTracker{i}, 'b', 'LineWidth', 0.5)
	hold on
end
hold off
xlabel('Time', 'fontsize', 14)
ylabel('$x$', 'interpreter', 'latex', 'fontsize', 16)
%title('Trajectories of Cells')
figname = 'Trajectories_all.png';
print(gcf, figname, '-dpng')
close

% plot trajectorires of n random cells
n_cells = 5;
idx_random = randperm(N, n_cells);

figure;
for i = 1:n_cells
	plot(tTracker{idx_random(i)}, xTracker{idx_random(i)}, 'LineWidth', 0.5)
	hold on
end
hold off
xlabel('Time', 'fontsize', 14)
ylabel('$x$', 'interpreter', 'latex', 'fontsize', 16)
%title('Trajectories of Cells')
figname = 'Trajectories_n.png';
print(gcf, figname, '-dpng')
close

% plot of the distribution at specific times
num_figs = 6;
timepoints = linspace(0, t_max, num_figs);

for i = 1:num_figs
	t_fig = timepoints(i);
	Px = zeros(size(x));
	list_X = zeros(1, N);

	for j = 1:N
		t_cell = tTracker{j};
		x_cell = xTracker{j};

		idx_valid = ~isnan(t_cell);
		t_cell = t_cell(idx_valid);
		x_cell = x_cell(idx_valid);

		idx_t_fig = find(t_cell >= t_fig, 1);
		if isempty(idx_t_fig)
			idx_t_fig = length(t_cell);
		end

		x_value = x_cell(idx_t_fig);
		[~, idx_x] = min(abs(x - x_value));
		Px(idx_x) = Px(idx_x) + 1;
		list_X(j) = x(idx_x);
	end

	Px = Px/Px(idx_x_bar);

	figure;
	plot(x, Px, 'LineWidth', 2)
	hold on
	plot(x, Pss_X(x), 'LineWidth', 2)
	hold off
	xlabel('$x$', 'interpreter', 'latex', 'fontsize', 16)
	ylabel('$X$', 'interpreter', 'latex', 'fontsize', 16, 'rotation', 0)
	%ylim([0 1.2])
	legend({'Stochastic', 'Airy'})
	%title(['Population Distribution, t = ' num2str(t_fig)])
	figname = ['Px_' num2str(i) '.png'];
	print(gcf, figname, '-dpng')
	close

	figure;
	histogram(list_X)
	xlabel('$x$', 'interpreter', 'latex', 'fontsize', 16)
	ylabel('Number of Cells', 'fontsize', 14)
	%title(['Histogram at t = ' num2str(t_fig)])
	figname = ['Histogram_' num2str(i) '.png'];
	print(gcf, figname, '-dpng')
	close

end

% time averaged distribution between t1 and t2
t1 = timepoints(end-1);
t2 = timepoints(end);
dt = 0.1;
t12 = t1:dt:t2;
Pxt = zeros(length(x), length(t12));

for i = 1:N

	t_cell = tTracker{i};
	x_cell = xTracker{i};

	idx_valid = ~isnan(t_cell);
	t_cell = t_cell(idx_valid);
	x_cell = x_cell(idx_valid);

	for j = 1:length(t12)
		t_j = t12(j);
		idx_t_j = find(t_cell >= t_j, 1);

		if isempty(idx_t_j)
			idx_t_j = length(t_cell);
		end

		x_value = x_cell(idx_t_j);
		[~, idx_x] = min(abs(x - x_value));
		Pxt(idx_x, j) = Pxt(idx_x, j) + 1;
	end

end

tavgPx = trapz(t12, Pxt, 2) / (t2 - t1);
tavgPx = tavgPx/tavgPx(idx_x_bar);

figure;
plot(x, tavgPx, 'LineWidth', 2)
hold on
plot(x, Pss_X(x), 'LineWidth', 2)
xlabel('$x$', 'interpreter', 'latex', 'fontsize', 16)
ylabel('$X$', 'interpreter', 'latex', 'fontsize', 16, 'rotation', 0)
%ylim([0 1.2])
%title(['Time-Averaged Distribution from t = ' num2str(t1) ' to t = ' num2str(t2)])
legend({'Stochastic', 'Airy'})
figname = 'tavgPx.png';
print(gcf, figname, '-dpng')
close


	

	




			

