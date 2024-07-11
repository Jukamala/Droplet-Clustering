import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans as KMeans
import ot
from scipy.sparse.csgraph import floyd_warshall
from latent_model import get_full_latent_by_time
from tools import rgb_to_hsv
from data import min_max_normalized

bbox = None

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Justus\\PycharmProjects\\Droplet-Clustering\\venv\\ffmpeg.exe'
sns.set_theme(style='whitegrid')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes3d.xaxis.panecolor'] = [0.98, 0.98, 0.98]
plt.rcParams['axes3d.yaxis.panecolor'] = [0.98, 0.98, 0.98]
plt.rcParams['axes3d.zaxis.panecolor'] = [0.98, 0.98, 0.98]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
textwidth = 5.50107


"""
Contains methods to visualize the latent space (and its clustering),
both at fixed time steps and animated over all time steps.
"""


def plot_latent_space(data, model, gmm=None, decoder=None, k=5, d=3, t_plot=0, plot=None, animate=None,
                      final=False, savepresuf=('tmp_', ''), latent_type='vae'):
    """
    Visualize the latent space

    :param data: Dataloader without time-mixed batches
    :param model: (VAE) latent encoder
    :param gmm: Fitted cluster object
    :param k: number of clusters
    :param d: number of latent dimensions
    :param plot, animate - list: kinds of plots to be visualized, from
        'latent3D' | 'latent3D-cluster': Encoded data, colors indicate continuous RGB | cluster
        'latent2D' | 'latent2D-cluster': 3 Projections of the above
        'data3D' | 'data3D-cluster': Original data
        'data1D': Original data, sorted colors by height
        'decoded-fixed': Decoded data, at fixed points
        'mass3D': Plot mass per point
    :param t_plot: time at which to plot (last time is -1)
    :param final: if True, save as svg without title
    :param latent_type: change color normalization and hue shift based on latent space
    """
    if plot is None and animate is None:
        return
    assert latent_type in ['vae', 'mom']
    # todo: assert cluster needs gmm
    # Prepare figures
    fig_ax = {}
    for task in (set(plot) if plot is not None else set()) | (set(animate) if animate is not None else set()):
        if '3D' in task:
            # scale up x1.5 for finer scatter plots
            fig = plt.figure(figsize=(1.5 * textwidth, 1.5 * 0.9 * textwidth))
            ax = fig.add_subplot(111, projection='3d')
            if 'latent' in task:
                if latent_type == 'mom':
                    l = np.array([0, 0, -50])
                    u = np.array([25, 10, 50])
                else:
                    l = np.array([-1.5, -1, -2])
                    u = np.array([1, 1.5, 1.5])
            elif 'decoded' in task:
                l = np.array([0, 0, -1e-6])
                u = np.array([32, 32, 1e-4])
            else:
                l = np.array([0, 0, 0])
                u = np.array([639, 639, 74])
            # u = np.array([2, 2, 2]) if 'latent' in task else np.array([511, 511, 149])
            # adjust spacing
            if not final:
                ax.set_title('t')
            if latent_type == 'mom':
                pass
            else:
                ax.set_xticks([-1, 0, 1])
                ax.set_yticks([-1, 0, 1])
                ax.set_zticks([-1, 0, 1])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            fig.tight_layout()

            def reset(l=l, u=u, task=task):
                if 'all' not in task:
                    ax.clear()
                if l is not None:
                    ax.set_xlim(l[0], u[0])
                    ax.set_ylim(l[1], u[1])
                    ax.set_zlim(l[2], u[2])

        elif '2D' in task:
            c = 1 if not final else 2.5
            fig, ax = plt.subplots(1, 3, figsize=(c * textwidth, 0.35 * c * textwidth))
            if latent_type == 'mom':
                l = np.array([0, 0, -50]) if 'latent' in task else np.array([0, 0, 0])
                u = np.array([25, 10, 50]) if 'latent' in task else np.array([639, 639, 74])
            else:
                l = np.array([-2, -2, -2]) if 'latent' in task else np.array([0, 0, 0])
                u = np.array([2, 2, 2]) if 'latent' in task else np.array([639, 639, 74])

            # adjust spacing
            if not final:
                ax[1].set_title('t')
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('y')
            ax[1].set_xlabel('x')
            ax[1].set_ylabel('y')
            ax[2].set_xlabel('x')
            ax[2].set_ylabel('y')
            fig.tight_layout()

            def reset(l=l, u=u):
                ax[0].clear()
                ax[1].clear()
                ax[2].clear()
                if l is not None:
                    ax[0].set_xlim(l[0], u[0])
                    ax[0].set_ylim(l[1], u[1])
                    ax[1].set_xlim(l[0], u[0])
                    ax[1].set_ylim(l[2], u[2])
                    ax[2].set_xlim(l[1], u[1])
                    ax[2].set_ylim(l[2], u[2])

        elif '1D' in task:
            fig, ax = plt.subplots(figsize=(textwidth, textwidth))

            # adjust spacing
            if not final:
                ax.set_title('t')
            ax.set_ylabel('y')
            fig.tight_layout()

            def reset():
                ax.clear()
                ax.set_ylim(0, 50)

        elif 'decoded' in task:
            fig, ax = plt.subplots(1, 2, figsize=(textwidth, 0.5 * textwidth))
            if not final:
                ax[0].set_title('t')
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('y')
            ax[1].set_xlabel('x')
            ax[1].set_ylabel('y')
            ax[1].set_title('t')
            fig.tight_layout()

            def reset():
                ax[0].clear()
                ax[1].clear()
                sns.despine(left=True, bottom=True)
                if not final:
                    ax[0].set_title('Latent Space Neighbors before Encoding')
                ax[0].set_xlabel('Droplet size')
                ax[0].set_ylabel('Mixing ratio kg/kg')
                ax[1].set_xlabel('Droplet size')
                ax[1].set_ylabel('Mixing ratio (normalized)')
                if not final:
                    ax[1].set_title('Decoded Latent Space Neighbors')
                # ax[0].set_ylim(-0.03, 0.43)
                ax[0].set_ylim(-0.000001, 0.00006)
                ax[1].set_ylim(-0.03, 0.43)
                fig.tight_layout()
        else:
            raise ValueError('Unknown plot task')

        fig_ax[task] = (fig, ax, reset)

    if latent_type == 'mom':
        latent_pl_args = dict(s=1, alpha=1/10)
    else:
        latent_pl_args = dict(s=1, alpha=1/255)
    cluster_pl_args = dict(cmap='tab10', vmin=0, vmax=9, s=1, alpha=1/255)
    center_pl_args = dict(c=list(range(k)), cmap='tab10', s=25, marker='X', vmin=0, vmax=9)
    mass_pl_args = dict(cmap='Blues', alpha=1/255)
    mass_pl2_args = dict(cmap='Blues')

    # Pathway of precipitation
    if latent_type == "mom":
        fixed = None
    else:
        fixed = np.array([
            [0.2, 0, 0],         # red
            [0.5, 0.6, 0],       # yellow
            [-0.25, 1, 0],
            [-1, 0.5, 0],
            [-1.5, 0, -0.3],     # green
            [-1, -0.4, 0.5],
            [-0.8, -0.2, 1.2],
            [-0.2, 0.5, 1.6],    # blue
            [0.2, -0.5, 0.5],    # magenta
            [0, 0, -0.5],        # orange
            [-0.5, 0, -1],       # olive
        ])

    # Start recording
    if animate is not None:
        writers = {}
        for task in animate:
            writers[task] = FFMpegWriter(fps=5)
            writers[task].setup(fig_ax[task][0], 'results/%s%s%s.mp4' % (savepresuf[0], task, savepresuf[1]))

    if gmm is not None:
        means, stds = gmm.model_.means, gmm.model_.covariances
        # means = km.model_.centroids

    old_latent, old_x, old_y, old_z, old_mass = None, None, None, None, None
    km_old = None
    for latent, labels, x, y, z, t, mass in get_full_latent_by_time(data, model, gmm):
        ths_tasks = (set(plot) if plot is not None and t_plot == t else set()) |\
                    (set(animate) if animate is not None else set()) |\
                    (set([p for p in plot if 'all' in p]) if plot is not None else set())

        # Normalize and compute colors
        if latent_type == 'mom':
            # mn, mx = np.array([4.5, 11, 17]), np.array([17, 55, 90])  # raw normalized moments
            mn, mx = np.array([4.5, 0, -15]), np.array([15, 4, 15])  # m + sigma + skew
            # print(np.percentile(latent, 1, axis=0), np.percentile(latent, 99, axis=0))
            # mn, mx = np.percentile(latent, 3, axis=0), np.percentile(latent, 97, axis=0)
        else:
            # mn, mx = np.array([-1.4, -1.5, -0.85]), np.array([0.78, 1.2, 1.8])  # old
            mn, mx = np.array([-1.2, -0.95, -1.15]), np.array([0.1, 0.44, 0.95])  # paper

        latent_cols = min_max_normalized(latent, mn=mn, mx=mx)
        if np.any(['mass' in task for task in ths_tasks]):
            mass_cols_xy = torch.zeros(640, 640)
            mass_cols_xz = torch.zeros(640, 75)
            mass_cols_yz = torch.zeros(640, 75)
            for x_, y_, z_, m_ in zip(x.to(int), y.to(int), z.to(int), mass):
                mass_cols_xy[x_, y_] += m_
                mass_cols_xz[x_, z_] += m_
                mass_cols_yz[y_, z_] += m_
            mass_cols_xy = min_max_normalized(mass_cols_xy, mn=1e-5, mx=1e-2, u=4)
            mass_cols_xz = min_max_normalized(mass_cols_xz, mn=2e-5, mx=0.15, u=4)
            mass_cols_yz = min_max_normalized(mass_cols_yz, mn=2e-5, mx=0.15, u=4)
            mass_cols = min_max_normalized(mass, mn=1e-5, mx=1e-3, u=4)

        # Re-adjust fixed points to nearby data and compute neighbors
        if t == (t_plot if not t_plot == -1 else 25) and fixed is not None:
            dist = torch.sum((latent[:, None] - fixed[None, :])**2, axis=2)
            nearest = torch.argsort(dist, dim=0)[:1000]
            fixed = np.array(torch.stack([latent[nearest[:, i]].mean(dim=0) for i in range(fixed.shape[0])]))
            fixed_cols = min_max_normalized(fixed, mn=mn, mx=mx)

        # max number of points in latent plots 3D/2D (instead of all ~2.500.000)
        n3, n2 = 500000, 500000
        n3a = 0 if t % 47 < 4 else 10000 if t % 47 < 23 else 50000

        # in latent space
        for task in {'latent3D', 'latent3D-all', 'latent3D-cluster',
                     'latent3D-fixed', 'latent3D-fixed-all', 'latent3D-mass'} & ths_tasks:
            fig, ax, reset = fig_ax[task]
            reset()
            if task == 'latent3D':
                ax.scatter(latent[:n3, 0], latent[:n3, 1], latent[:n3, 2], c=latent_cols[:n3], **latent_pl_args)
            elif task == 'latent3D-all':
                ax.scatter(latent[:n3a, 0], latent[:n3a, 1], latent[:n3a, 2], c=latent_cols[:n3a], **latent_pl_args)
                ax.set_box_aspect([4, 4, 3.8])
                if t == -1 and fixed is not None:
                    # ax.plot(fixed[:7, 0], fixed[:7, 1], fixed[:7, 2], lw=3, c='grey')
                    path = np.copy(fixed[:8])
                    path[-1] = 0.05 * path[-2] + 0.95 * path[-1]
                    ax.plot(path[:, 0], path[:, 1], path[:, 2], lw=2, c='red')
                    # Arrowhead

                    class Arrow3D(FancyArrowPatch):
                        def __init__(self, xs, ys, zs, *args, **kwargs):
                            super().__init__((0, 0), (0, 0), *args, **kwargs)
                            self._verts3d = xs, ys, zs

                        def do_3d_projection(self, renderer=None):
                            xs3d, ys3d, zs3d = self._verts3d
                            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
                            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

                            return 1000

                    ax.add_artist(Arrow3D(fixed[6:8, 0], fixed[6:8, 1], fixed[6:8, 2], arrowstyle="-|>", color="red", mutation_scale=20, shrinkA=0, shrinkB=0))
            elif task == 'latent3D-fixed':
                ax.scatter(latent[:n3, 0], latent[:n3, 1], latent[:n3, 2], c=latent_cols[:n3], **latent_pl_args)
                # Fixed points
                ax.scatter(fixed[:, 0], fixed[:, 1], fixed[:, 2], zorder=1000, c='black', marker='X', s=35)
                ax.scatter(fixed[:, 0], fixed[:, 1], fixed[:, 2], zorder=1100, c=fixed_cols, marker='X', s=25)
            elif task == 'latent3D-fixed-all':
                ax.scatter(latent[:n3a, 0], latent[:n3a, 1], latent[:n3a, 2], c=latent_cols[:n3a], **latent_pl_args)
                if t == -1:
                    # Fixed line
                    ax.plot(fixed[:7, 0], fixed[:7, 1], fixed[:7, 2], c='black', lw=2)
                    # Fixed plots
                    ax.scatter(fixed[:7, 0], fixed[:7, 1], fixed[:7, 2], zorder=1000, c='black', marker='X', s=25)
                    ax.scatter(fixed[:7, 0], fixed[:7, 1], fixed[:7, 2], zorder=1100, c=fixed_cols[:7], marker='X', s=20)
            elif task == 'latent3D-cluster':
                ax.scatter(latent[:n3, 0], latent[:n3, 1], latent[:n3, 2], c=labels[:n3], **cluster_pl_args)
                # Cluster centers
                ax.scatter(means[:, 0], means[:, 1], means[:, 2], **center_pl_args)
            elif task == 'latent3D-mass':
                ax.scatter(latent[:n3, 0], latent[:n3, 1], latent[:n3, 2], s=mass_cols[:n3], c=mass_cols[:n3], **mass_pl_args)
                # ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], s=mass_cols, c=mass_cols, **mass_pl_args)
            else:
                raise ValueError('Unknown task')
            if not final:
                ax.set_title('Latent Space Encoding [ %dh %d0m ]' % ((t + 1)//6, (t+1) % 6))
            if latent_type == 'mom':
                ax.set_xlabel('mean')
                ax.set_ylabel('standard deviation')
                ax.set_zlabel('skewness')
            else:
                ax.set_xlabel('1st latent dim [red]')
                ax.set_ylabel('2nd latent dim [green]')
                ax.set_zlabel('3rd latent dim [blue]')
            if plot is not None and task in plot and t_plot == t:
                fig_ax[task][0].savefig('results/%s%s%s.png' % (savepresuf[0], task, savepresuf[1]))
                # writers[task].setup(fig_ax[task][0], 'results/%s%s%s.png' % (savepresuf[0], task, savepresuf[1]))
            if animate is not None and task in animate:
                writers[task].grab_frame()

        # in projected latent space
        for task in {'latent2D', 'latent2D-all', 'latent2D-cluster', 'latent2D-fixed'} & ths_tasks:
            fig, ax, reset = fig_ax[task]
            reset()
            if task == 'latent2D':
                ax[0].scatter(latent[:n2, 0], latent[:n2, 1], c=latent_cols[:n2], **latent_pl_args)
                ax[1].scatter(latent[:n2, 0], latent[:n2, 2], c=latent_cols[:n2], **latent_pl_args)
                ax[2].scatter(latent[:n2, 1], latent[:n2, 2], c=latent_cols[:n2], **latent_pl_args)
            elif task == 'latent2D-all':
                ax[0].scatter(latent[:n3a, 0], latent[:n3a, 1], c=latent_cols[:n3a], **latent_pl_args)
                ax[1].scatter(latent[:n3a, 0], latent[:n3a, 2], c=latent_cols[:n3a], **latent_pl_args)
                ax[2].scatter(latent[:n3a, 1], latent[:n3a, 2], c=latent_cols[:n3a], **latent_pl_args)
                if t == -1:
                    ax[0].plot(fixed[:7, 0], fixed[:7, 1], lw=3, c='grey')
                    ax[0].plot(fixed[:7, 0], fixed[:7, 1], lw=1.5, c='red')
                    ax[1].plot(fixed[:7, 0], fixed[:7, 2], lw=3, c='grey')
                    ax[1].plot(fixed[:7, 0], fixed[:7, 2], lw=1.5, c='red')
                    ax[2].plot(fixed[:7, 1], fixed[:7, 2], lw=3, c='grey')
                    ax[2].plot(fixed[:7, 1], fixed[:7, 2], lw=1.5, c='red')
            elif task == 'latent2D-fixed':
                ax[0].scatter(latent[:n2, 0], latent[:n2, 1], c=latent_cols[:n2], **latent_pl_args)
                ax[1].scatter(latent[:n2, 0], latent[:n2, 2], c=latent_cols[:n2], **latent_pl_args)
                ax[2].scatter(latent[:n2, 1], latent[:n2, 2], c=latent_cols[:n2], **latent_pl_args)
                # Fixed points
                ax[0].scatter(fixed[:, 0], fixed[:, 1], c='black', s=35, marker='X')
                ax[1].scatter(fixed[:, 0], fixed[:, 2], c='black', s=35, marker='X')
                ax[2].scatter(fixed[:, 1], fixed[:, 2], c='black', s=35, marker='X')
                ax[0].scatter(fixed[:, 0], fixed[:, 1], c=fixed_cols, marker='X', s=25)
                ax[1].scatter(fixed[:, 0], fixed[:, 2], c=fixed_cols, marker='X', s=25)
                ax[2].scatter(fixed[:, 1], fixed[:, 2], c=fixed_cols, marker='X', s=25)
            elif task == 'latent2D-cluster':
                ax[0].scatter(latent[:n2, 0], latent[:n2, 1], s=1, c=labels[:n2], **cluster_pl_args)
                ax[1].scatter(latent[:n2, 0], latent[:n2, 2], s=1, c=labels[:n2], **cluster_pl_args)
                ax[2].scatter(latent[:n2, 1], latent[:n2, 2], s=1, c=labels[:n2], **cluster_pl_args)
                # Cluster centers
                ax[0].scatter(means[:, 0], means[:, 1], c='black', s=30, marker='X')
                ax[1].scatter(means[:, 0], means[:, 2], c='black', s=30, marker='X')
                ax[2].scatter(means[:, 1], means[:, 2], c='black', s=30, marker='X')
                ax[0].scatter(means[:, 0], means[:, 1], **center_pl_args)
                ax[1].scatter(means[:, 0], means[:, 2], **center_pl_args)
                ax[2].scatter(means[:, 1], means[:, 2], **center_pl_args)
            else:
                raise ValueError('Unknown task')
            if not final:
                ax[1].set_title('Latent Space Encoding [ %dh %d0m ]' % ((t + 1)//6, (t+1) % 6))
            if latent_type == 'mom':
                ax[0].set_xlabel('mean')
                ax[0].set_ylabel('standard deviation')
                ax[1].set_xlabel('mean')
                ax[1].set_ylabel('skewness')
                ax[2].set_xlabel('standard deviation')
                ax[2].set_ylabel('skewness')
            else:
                ax[0].set_xlabel('1st dim [red]')
                ax[0].set_ylabel('2nd dim')
                ax[1].set_xlabel('1st dim')
                ax[1].set_ylabel('3th dim')
                ax[2].set_xlabel('2nd dim')
                ax[2].set_ylabel('3th dim')
            if plot is not None and task in plot and t_plot == t:
                fig_ax[task][0].savefig('results/%s%s%s.png' % (savepresuf[0], task, savepresuf[1]))
            if animate is not None and task in animate:
                writers[task].grab_frame()

        # in data space
        for task in {'data3D', 'data3D-cluster', 'mass3D'} & ths_tasks:
            fig, ax, reset = fig_ax[task]
            reset()
            if task == 'data3D':
                ax.scatter(x, y, z, c=latent_cols, **latent_pl_args)
                # ax.scatter(x, y, z, c='C0', **latent_pl_args)
            elif task == 'data3D-cluster':
                ax.scatter(x, y, z, c=labels, **cluster_pl_args)
            elif task == 'mass3D':
                ax.scatter(x, y, z, s=mass_cols, c=mass_cols, **mass_pl_args)
            else:
                raise ValueError('Unknown task')
            if not final:
                ax.set_title('[ %dh %d0m ]' % ((t + 1)//6, (t+1) % 6))
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('height')
            if plot is not None and task in plot and t_plot == t:
                fig_ax[task][0].savefig('results/%s%s%s.png' % (savepresuf[0], task, savepresuf[1]))
            if animate is not None and task in animate:
                writers[task].grab_frame()

        # in projected (2D) data space
        for task in {'mass2D'} & ths_tasks:
            fig, ax, reset = fig_ax[task]
            reset()
            if task == 'mass2D':
                x_, y_, z_ = [np.arange(s) for s in [640, 640, 75]]
                ax[0].scatter(*np.meshgrid(x_, y_, indexing='ij'), s=mass_cols_xy, c=mass_cols_xy, **mass_pl2_args)
                ax[1].scatter(*np.meshgrid(x_, z_, indexing='ij'), s=mass_cols_xz, c=mass_cols_xz, **mass_pl2_args)
                ax[2].scatter(*np.meshgrid(y_, z_, indexing='ij'), s=mass_cols_yz, c=mass_cols_yz, **mass_pl2_args)
            else:
                raise ValueError('Unknown task')
            if not final:
                ax[1].set_title('[ %dh %d0m ]' % ((t + 1)//6, (t+1) % 6))
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('y')
            ax[1].set_xlabel('x')
            ax[1].set_ylabel('height')
            ax[2].set_xlabel('y')
            ax[2].set_ylabel('height')
            if plot is not None and task in plot and t_plot == t:
                fig_ax[task][0].savefig('results/%s%s%s.png' % (savepresuf[0], task, savepresuf[1]))
            if animate is not None and task in animate:
                writers[task].grab_frame()

        # in projected (1D) and sorted data space
        for task in {'data1D', 'data1D-cluster'} & ths_tasks:
            fig, ax, reset = fig_ax[task]
            reset()
            if task == 'data1D':
                height = 150
                sorted = np.ones((height, 500, 3))
                for h in range(height):
                    # Sort colors by hue
                    rgb = latent_cols[z == h]
                    rgb = rgb[rgb.std(axis=1) > 0]
                    # remove white/black
                    if rgb.shape[0] < 100:
                        continue
                    hsv = rgb_to_hsv(rgb)
                    # sample 500
                    idxs = np.linspace(0, rgb.shape[0]-1, 500).astype(int)
                    ths_sorted = rgb[np.argsort(hsv[:, 0])][idxs]
                    # set saturation to 0.7 and value to 1
                    mn = ths_sorted.min(axis=1)[0][:, None]
                    mx = ths_sorted.max(axis=1)[0][:, None]
                    sorted[h] = 0.3 + 0.7 * (ths_sorted - mn) / (mx - mn)
                # saturation and value set to 1
                with sns.axes_style("white"):
                    ax.imshow(sorted, aspect='auto', origin='lower', interpolation='bilinear')
            else:
                raise NotImplementedError
            if not final:
                ax.set_ylabel('height')
                ax.set_title('[ %dh %d0m ]' % ((t + 1) // 6, (t + 1) % 6))
            else:
                # left panel
                if t_plot == 11:
                    for item in ([ax.yaxis.label] + ax.get_yticklabels()):
                        item.set_fontsize(20)
                else:
                    ax.set_yticklabels([])
                    ax.set_ylabel('')
            sns.despine(bottom=True)
            ax.set_xticks([])
            if plot is not None and task in plot and t_plot == t:
                global bbox
                if bbox is None:
                    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
                fig_ax[task][0].savefig('results/%s%s%s.png' % (savepresuf[0], task, savepresuf[1]), bbox_inches=bbox, pad_inches=0)
                # fig_ax[task][0].savefig('results/%s%s%s.png' % (savepresuf[0], task, savepresuf[1]))
            if animate is not None and task in animate:
                writers[task].grab_frame()

        # Differences in latent space
        for task in {'latent3D-trace', 'latent3D-change'} & ths_tasks:
            fig, ax, reset = fig_ax[task]
            reset()

            # Sequential clustering
            n = 1000
            if old_latent is None:
                continue
            if km_old is None:
                km_old = KMeans(n_clusters=n, random_state=0, init='k-means++', n_init=1, batch_size=25000)
                km_old.fit(old_latent, sample_weight=old_mass)
                c_old = km_old.cluster_centers_
                m_old = np.array([old_mass[km_old.labels_ == i].sum() for i in range(n)])
                # m_old /= m_old.sum()
                c_old = np.append(c_old, [[0, 1, 2]], axis=0)
                m_old = np.append(m_old, [0], axis=0)

            km = KMeans(n_clusters=n, random_state=0, init=km_old.cluster_centers_, n_init=1, batch_size=25000)
            km.fit(latent, sample_weight=mass)
            c = km.cluster_centers_
            m = np.array([mass[km.labels_ == i].sum() for i in range(n)])
            # m /= m.sum()
            c = np.append(c, [[0, 1, 2]], axis=0)
            m = np.append(m, [0], axis=0)
            if m_old.sum() > m.sum():
                m[-1] = m_old.sum() - m.sum()
            else:
                m_old[-1] = m.sum() - m_old.sum()

            if task == 'latent3D-change':
                # CLuster color is change in mass for next step
                labels = km_old.predict(latent, sample_weight=mass)
                m_new = np.array([mass[labels == i].sum() for i in range(n)])
                dif = m_new - m_old[:-1]
                v = min([np.abs(np.percentile(dif, q)) for q in [3, 97]])
                ax.scatter(c_old[:-1, 0], c_old[:-1, 1], c_old[:-1, 2], c=dif, vmin=-v, vmax=v, cmap='coolwarm', s=3)
            elif task == 'latent3D-trace':
                # Optimal transport in data space
                # d1, d2 = old_x.shape[0], x.shape[0]
                # d = max(d1, d2)
                # p1, p2 = torch.zeros(d, 3), torch.zeros(d, 3)
                # m1, m2 = torch.zeros(d), torch.zeros(d)
                # p1[:d1, :] = torch.stack([old_x, old_y, old_z], dim=-1)
                # p2[:d2, :] = torch.stack([x, y, z], dim=-1)
                # m1[:d1] = old_mass
                # m2[:d2] = mass
                # C = ot.dist(p1, p2)
                # transport = ot.emd(m1, m2, C)

                # Optimal transport
                D_c = ot.dist(c_old, c)
                # otp = ot.emd(m_old, m, D_c)
                # Allow for stops c_old_i -> c_k -..-> c_j and sum segments
                D = ot.dist(c)
                D_s, pre = floyd_warshall(D, return_predecessors=True)
                D_cs = D_c[..., None] + D_s[None, ...]
                pre_c = np.argmin(D_cs, axis=1)
                D_c_full = np.take_along_axis(D_cs, np.expand_dims(pre_c, axis=1), axis=1).squeeze()

                # Option 1: c_old_i -> c_j
                otp = ot.emd(m_old, m, D_c_full)
                # Option 1: c_old_i -> c_k
                otp_full = np.zeros_like(otp)
                for i in range(n+1):
                    for j in range(n + 1):
                        otp_full[i, pre_c[i, j]] += otp[i, j]
                # todo: Option 3: accumulate c_old_i -> c_k -..-> c_j

                ax.scatter(c_old[:, 0], c_old[:, 1], c_old[:, 2], c='C0', s=1, alpha=0.3)
                ax.scatter(c[:, 0], c[:, 1], c[:, 2], c='C0', s=2)
                for i in range(n):
                    for j in range(n):
                        # Only plot relevant connections
                        rel = otp_full[i, j] / otp_full[:, j].sum()
                        if rel > 0.01:
                            rgb = np.array([c_old[i, 0] - c[j, 0], c_old[i, 1] - c[j, 1], c_old[i, 2] - c[j, 2]])
                            rgb = np.abs(rgb / np.linalg.norm(rgb))
                            ax.plot(np.array([c_old[i, 0], c[j, 0]]), np.array([c_old[i, 1], c[j, 1]]),
                                    np.array([c_old[i, 2], c[j, 2]]), c=rgb, lw=1, alpha=0.7 * rel)
            c_old = c
            m_old = m
            m_old[-1] = 0
            km_old = km

            ax.set_title('Latent Space Cluster Centers [ %dh %d0m ]' % ((t + 1) // 6, (t + 1) % 6))
            ax.set_xlabel('1st dim')
            ax.set_ylabel('2nd dim')
            ax.set_zlabel('3rd dim')
            if plot is not None and task in plot and t_plot == t:
                fig_ax[task][0].savefig('results/%s%s%s.png' % (savepresuf[0], task, savepresuf[1]), dpi=1000)
            if animate is not None and task in animate:
                writers[task].grab_frame()

        # fixed points in original data space, time as dimension
        for task in {'decoded3D-fixed-pre', 'decoded3D-fixed-post'} & ths_tasks:
            # animate only one time step
            fig, ax, reset = fig_ax[task]
            reset()

            fixed_ = torch.tensor(np.concatenate(
                [np.linspace(fixed[i], fixed[i + 1], 10)[:(9 if i < 5 else 10)] for i in range(7)]))
            fixed_cols = min_max_normalized(fixed_, mn=mn, mx=mx)

            ns = 1000
            nf = fixed_.shape[0]
            dist = torch.sum((latent[:, None] - fixed_[None, :]) ** 2, axis=2)
            nearest = torch.argsort(dist, dim=0)[:ns]
            dec = decoder(fixed_.to(device)).cpu()

            neighbors = torch.stack([data.dataset.__getitem__(nearest[:, i].sort()[0], fid=t)['x'] for i in range(nf)])
            neighbors_mean = neighbors.mean(axis=1)
            dec_neighbors = torch.stack([decoder(latent[nearest[:, i]].to(device)).cpu() for i in range(nf)])

            nt, nb = neighbors_mean.shape
            x_, y_ = np.meshgrid(np.arange(nb), np.arange(nt), indexing='ij')
            y_ = y_ * nb/nt
            if task == 'decoded3D-fixed-pre':
                ax.plot_surface(x_, y_, neighbors_mean.T, facecolors=np.tile(fixed_cols[None, :], [nb, 1, 1]),
                                edgecolor='k', shade=False)
            elif task == 'decoded3D-fixed-post':
                pass  # ax.plot(np.arange(33), dec_neighbors[k], alpha=1 / 100, c=neighbors_cols[k])
            else:
                raise ValueError('Unknown task')
            if not final:
                ax.set_title('[ %dh %d0m ]' % ((t + 1) // 6, (t + 1) % 6))
            ax.set_xlabel('Droplet size')
            ax.set_ylabel('Time')
            # ax.yaxis.set_major_formatter(lambda t, pos: '%dh %d0m' % ((t + 1) // 6, (t + 1) % 6))
            ax.set_yticks([])
            ax.set_zlabel('Mixing ratio [kg/kg]')

            if plot is not None and task in plot and t_plot == t:
                fig_ax[task][0].savefig('results/%s%s%s.png' % (savepresuf[0], task, savepresuf[1]))
            if animate is not None and task in animate:
                writers[task].grab_frame()

        old_latent, old_x, old_y, old_z, old_mass = latent, x, y, z, mass

    if animate is not None:
        for w in writers.values():
            w.finish()


def plot_components(gmm, decoder):
    centroids = decoder(gmm.model_.means.to(device)).detach().to('cpu')

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(gmm.num_components):
        lt = norm(loc=gmm.model_.means[k], scale=torch.sqrt(gmm.model_.covariances[k])).rvs((5000, gmm.model_.means.shape[1]))
        fs = decoder(torch.tensor(torch.tensor(lt).to(device).to(torch.float32))).detach().to('cpu')
        l, u = np.percentile(fs, 5, axis=0), np.percentile(fs, 95, axis=0)
        ax.fill_between(np.arange(0, 33), l, u, alpha=0.2, color="C%d" % k)
        # ax.plot(fs.T, color="C%d" % k, alpha=0.01)
    ax.plot(centroids.T)
    plt.savefig('results/tmp0.png')
    plt.show()
