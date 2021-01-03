function [xdata,ydata]=get_figure_data(fig_number)

figure(fig_number);

h=get(gca,'Children');
xdata=get(h,'XData');
ydata=get(h,'YData');
