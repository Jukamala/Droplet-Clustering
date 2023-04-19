import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns
from scipy.stats import norm
from gmm_torch import get_full_latent_by_time
from tools import rgb_to_hsv

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\Justus\\PycharmProjects\\Tests\\venv\\ffmpeg.exe'
sns.set_theme(style='whitegrid')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


"""
Contains methods to visualize the latent space (and its clustering),
both at fixed time steps and animated over all time steps.
"""


def plot_latent_space(data, model, gmm=None, decoder=None, k=5, d=3, t_plot=0, plot=None, animate=None):
    """
    Visualize the latent space

    :param data: Dataloader without time-mixed batches
    :param model: VAE latent encoder (of the mean)
    :param gmm: Fitted cluster object
    :param k: number of clusters
    :param d: number of latent dimensions
    :param plot, animate - list: kinds of plots to be visualized, from
        'latent3D' | 'latent3D-cluster': Encoded data, colors indicate continuous RGB | cluster
        'latent2D' | 'latent2D-cluster': 3 Projections of the above
        'data3D' | 'data3D-cluster': Original data
        'data1D': Original data, sorted colors by height
        'decoded-fixed': Decoded data, at fixed points
    :param t_plot: time at which to
    """
    # todo: assert cluster needs gmm
    # Prepare figures
    fig_ax = {}
    for task in (set(plot) if plot is not None else set()) | (set(animate) if animate is not None else set()):
        if '3D' in task:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            l = np.array([-2, -2, -2]) if 'latent' in task else np.array([0, 0, 0])
            u = np.array([2, 2, 2]) if 'latent' in task else np.array([639, 639, 74])
            # adjust spacing
            ax.set_title('t')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            fig.tight_layout()

            def reset(l=l, u=u):
                ax.clear()
                ax.set_xlim(l[0], u[0])
                ax.set_ylim(l[1], u[1])
                ax.set_zlim(l[2], u[2])

        elif '2D' in task:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            # adjust spacing
            ax[1].set_title('t')
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('y')
            ax[1].set_xlabel('x')
            ax[1].set_ylabel('y')
            ax[2].set_xlabel('x')
            ax[2].set_ylabel('y')
            fig.tight_layout()

            def reset():
                ax[0].clear()
                ax[1].clear()
                ax[2].clear()
                ax[0].set_xlim(-2, 2)
                ax[0].set_ylim(-2, 2)
                ax[1].set_xlim(-2, 2)
                ax[1].set_ylim(-2, 2)
                ax[2].set_xlim(-2, 2)
                ax[2].set_ylim(-2, 2)

        elif '1D' in task:
            fig, ax = plt.subplots(figsize=(8, 8))
            # adjust spacing
            ax.set_title('t')
            ax.set_ylabel('y')
            fig.tight_layout()
            reset = ax.clear

        elif 'decoded' in task:
            fig, ax = plt.subplots(1, 2, figsize=(10, 6))
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
                ax[0].set_title('Latent Space Neighbors before Encoding')
                ax[0].set_xlabel('Droplet size')
                ax[0].set_ylabel('Mixing ratio kg/kg')
                ax[1].set_xlabel('Droplet size')
                ax[1].set_ylabel('Mixing ratio (normalized)')
                ax[1].set_title('Decoded Latent Space Neighbors')
                # ax[0].set_ylim(-0.03, 0.43)
                ax[0].set_ylim(-0.000001, 0.00006)
                ax[1].set_ylim(-0.03, 0.43)
                fig.tight_layout()
        else:
            raise ValueError('Unknown plot task')

        fig_ax[task] = (fig, ax, reset)

    latent_pl_args = dict(s=1, alpha=1/255)
    cluster_pl_args = dict(cmap='tab10', vmin=0, vmax=9, s=1, alpha=1/255)
    center_pl_args = dict(c=list(range(k)), cmap='tab10', s=25, marker='X', vmin=0, vmax=9)

    fixed = np.array([
        [0.5, 0.6, 0],       # yellow
        [-0.25, 1, 0],
        [-1, 0.5, 0],
        [-1.5, 0, -0.3],     # green
        [-1, -0.4, 0.5],
        [-0.8, -0.2, 1.2],
        [-0.2, 0.5, 1.3],    # blue
        [0.2, -0.5, 0.5],    # magenta
        [0, 0, -0.5],        # orange
        [-0.5, 0, -1],       # olive
    ])

    # Start recording
    if animate is not None:
        writers = {}
        for task in animate:
            writers[task] = FFMpegWriter(fps=5)
            writers[task].setup(fig_ax[task][0], 'results/tmp_%s.mp4' % task)

    if gmm is not None:
        means, stds = gmm.model_.means, gmm.model_.covariances
        # means = km.model_.centroids

    for latent, labels, x, y, z, t in get_full_latent_by_time(data, model, gmm):
        # atex, vae_full
        # mn, mx = np.array([-1.4, -1.5, -0.85]), np.array([0.78, 1.2, 1.8])
        # atex, vae_fullest_max
        mn, mx = np.array([-1.2, -0.95, -1.15]), np.array([0.1, 0.44, 0.95])
        # print(np.percentile(latent, 1, axis=0), np.percentile(latent, 99, axis=0))
        latent_cols = (np.clip(latent, mn, mx) - mn) / (mx - mn)

        if t == t_plot:
            # Re-adjust fixed points to nearby data and compute neighbors
            dist = torch.sum((latent[:, None] - fixed[None, :])**2, axis=2)
            nearest = torch.argsort(dist, dim=0)[:1000]
            fixed = torch.stack([latent[nearest[:, i]].mean(dim=0) for i in range(fixed.shape[0])])

        # max number of points in latent plots 3D/2D (instead of all ~2.500.000)
        n3, n2 = 500000, 500000

        # in latent space
        ths_tasks = (set(plot) if plot is not None and t_plot == t else set()) | (set(animate) if animate is not None else set())
        for task in {'latent3D', 'latent3D-cluster', 'latent3D-fixed'} & ths_tasks:
            fig, ax, reset = fig_ax[task]
            reset()
            if task == 'latent3D':
                ax.scatter(latent[:n3, 0], latent[:n3, 1], latent[:n3, 2], c=latent_cols[:n3], **latent_pl_args)
            elif task == 'latent3D-fixed':
                ax.scatter(latent[:n3, 0], latent[:n3, 1], latent[:n3, 2], c=latent_cols[:n3], **latent_pl_args)
                # Fixed points
                ax.scatter(fixed[:, 0], fixed[:, 1], fixed[:, 2], zorder=1000, c='black', marker='X', s=35)
                ax.scatter(fixed[:, 0], fixed[:, 1], fixed[:, 2], zorder=1100, c=fixed_cols, marker='X', s=25)
            else:
                ax.scatter(latent[:n3, 0], latent[:n3, 1], latent[:n3, 2], c=labels[:n3], **cluster_pl_args)
                # Cluster centers
                ax.scatter(means[:, 0], means[:, 1], means[:, 2], **center_pl_args)
            ax.set_title('Latent Space Encoding [ %dh %d0m ]' % ((t + 1)//6, (t+1) % 6))
            ax.set_xlabel('1st dim')
            ax.set_ylabel('2nd dim')
            ax.set_zlabel('3rd dim')
            if plot is not None and task in plot and t_plot == t:
                fig_ax[task][0].savefig('results/tmp_%s.png' % task)
            if animate is not None and task in animate:
                writers[task].grab_frame()

        # in projected latent space
        for task in {'latent2D', 'latent2D-cluster', 'latent2D-fixed'} & ths_tasks:
            fig, ax, reset = fig_ax[task]
            reset()
            if task == 'latent2D':
                ax[0].scatter(latent[:n2, 0], latent[:n2, 1], c=latent_cols[:n2], **latent_pl_args)
                ax[1].scatter(latent[:n2, 0], latent[:n2, 2], c=latent_cols[:n2], **latent_pl_args)
                ax[2].scatter(latent[:n2, 1], latent[:n2, 2], c=latent_cols[:n2], **latent_pl_args)
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
            else:
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
            ax[1].set_title('Latent Space Encoding [ %dh %d0m ]' % ((t + 1)//6, (t+1) % 6))
            ax[0].set_xlabel('1st dim')
            ax[0].set_ylabel('2nd dim')
            ax[1].set_xlabel('1st dim')
            ax[1].set_ylabel('3th dim')
            ax[2].set_xlabel('2nd dim')
            ax[2].set_ylabel('3th dim')
            if plot is not None and task in plot and t_plot == t:
                fig_ax[task][0].savefig('results/tmp_%s.png' % task)
            if animate is not None and task in animate:
                writers[task].grab_frame()

        # in data space
        for task in {'data3D', 'data3D-cluster'} & ths_tasks:
            fig, ax, reset = fig_ax[task]
            reset()
            if task == 'data3D':
                ax.scatter(x, y, z, c=latent_cols, **latent_pl_args)
            else:
                ax.scatter(x, y, z, c=labels, **cluster_pl_args)
            ax.set_title('[ %dh %d0m ]' % ((t + 1)//6, (t+1) % 6))
            ax.set_zlabel('height')
            if plot is not None and task in plot and t_plot == t:
                fig_ax[task][0].savefig('results/tmp_%s.png' % task)
            if animate is not None and task in animate:
                writers[task].grab_frame()

        # in projected and sorted data space
        for task in {'data1D', 'data1D-cluster'} & ths_tasks:
            fig, ax, reset = fig_ax[task]
            reset()
            if task == 'data1D':
                sorted = np.ones((75, 500, 3))
                for h in range(75):
                    # Sort colors by hue
                    rgb = latent_cols[z == h]
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
                    ax.imshow(sorted, aspect='auto', origin='lower')
            else:
                raise NotImplementedError
            ax.set_title('[ %dh %d0m ]' % ((t + 1) // 6, (t + 1) % 6))
            ax.set_ylabel('height')
            sns.despine(bottom=True)
            ax.set_xticks([])
            if plot is not None and task in plot and t_plot == t:
                fig_ax[task][0].savefig('results/tmp_%s.png' % task)
            if animate is not None and task in animate:
                writers[task].grab_frame()

        # fixed points in decoded and original data space
        for task in {'decoded', 'decoded-fixed'} & ths_tasks:
            # animate only one time step
            if t != t_plot:
                break
            fig, ax, reset = fig_ax[task]
            reset()

            if task == 'decoded-fixed':
                if animate is not None and task in animate:
                    fixed_ = torch.tensor(np.concatenate(
                        [np.linspace(fixed[i], fixed[i+1], 10)[:(9 if i < 5 else 10)] for i in range(6)]))
                    fixed_cols = (np.clip(fixed_, mn, mx) - mn) / (mx - mn)
                else:
                    fixed_ = fixed
                    fixed_cols = (np.clip(fixed, mn, mx) - mn) / (mx - mn)

                dist = torch.sum((latent[:, None] - fixed_[None, :]) ** 2, axis=2)

                # sdist = dist.sort(dim=0)[0]
                # [plt.plot(np.sqrt(sdist[:50000, i]), c='black', lw=2) for i, col in enumerate(fixed_cols) if i < 7]
                # [plt.plot(np.sqrt(sdist[:50000, i]), c=col.numpy()) for i, col in enumerate(fixed_cols) if i < 7]
                # plt.xlabel('i-th neighbor')
                # plt.ylabel('Euclidian Distance')
                # plt.title('Data Sparsity')
                # plt.tight_layout()
                # sns.despine(left=True, bottom=True)

                nearest = torch.argsort(dist, dim=0)[:1000]
                dec = decoder(fixed_.to(device)).cpu()
                for i, col in enumerate(fixed_cols):
                    neighbors = data.dataset.__getitem__(nearest[:, i].sort()[0], fid=t)['x']
                    # neighbors /= neighbors.sum(axis=-1)[:, None]
                    dec_neighbors = decoder(latent[nearest[:, i]].to(device)).cpu()
                    neighbors_cols = latent_cols[nearest[:, i]].numpy()
                    # todo: avoid loop with color cycler
                    for k in range(nearest.shape[0]):
                        ax[0].plot(np.arange(33), neighbors[k], alpha=1/100, c=neighbors_cols[k])
                        ax[1].plot(np.arange(33), dec_neighbors[k], alpha=1/100, c=neighbors_cols[k])
                    if animate is not None and task in animate:
                        ax[1].plot(dec[i], c=0.8 * col.numpy(), lw=2)
                        writers[task].grab_frame()
                        reset()
                if plot is not None and task in plot:
                    for i, col in enumerate(fixed_cols):
                        ax[1].plot(dec[i], c=0.8 * col.numpy(), lw=2)
            else:
                d = data.dataset.__getitem__(slice(0, 100000), fid=t)['x']
                d /= d.sum(axis=-1)[:, None]
                dec = decoder(latent[:100000].to(device)).cpu()
                for k in range(100000):
                    ax[0].plot(np.arange(33), d[k], alpha=1/255, c=latent_cols[k].numpy())
                    ax[1].plot(np.arange(33), dec[k], alpha=1/255, c=latent_cols[k].numpy())
                ax[0].set_title('Before Encoding')
                ax[1].set_title('After Decoding')

            if plot is not None and task in plot:
                fig_ax[task][0].savefig('results/tmp_%s.png' % task)

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
