from visualization import plot_latent_space, plot_components
from latent_model import latent_cluster, latent_moments
from data import DATA_ATEX_ALL, DATA_DYCOMS_ALL, Data

if __name__ == '__main__':
    # Data
    data = Data(DATA_ATEX_ALL).dataloader(batch_size=25000, weak_shuffle=True, num_workers=4)
    data_plot = Data(DATA_ATEX_ALL[0]).dataloader(batch_size=100000, by_time=True, t_plot=[11, 23, 35])
    # data_plot = Data(DATA_ATEX_ALL[0]).dataloader(batch_size=100000, by_time=True)
    # data_plot = Data(DATA_ATEX_ALL).dataloader(batch_size=100000, by_time=True)

    # Train via: latent_cluster(data, load_path='models/atex_mse.cp')
    # latent_encoder, latent_decoder, gmm = latent_cluster(load_path='models/atex_mse.cp')
    latent_encoder, latent_decoder = latent_moments()

    # plot_components(gmm, latent_decoder)
    plot_latent_space(data_plot, latent_encoder, decoder=latent_decoder, gmm=None,
        # testing
        # t_plot=25, final=False,
        # plot=['latent3D', 'latent2D', 'data3D', 'data1D']
        # plot=['decoded-fixed', 'latent3D-fixed', 'latent2D-fixed']
        # plot=['mass2D', 'mass3D'],
        # plot=['latent3D-mass'],
        # plot=['latent3D-trace'],
        # plot=['latent3D-change'],
        # animate=['latent3D-mass'],
        # animate=['latent3D-trace'],
        # animate=['latent3D-change'],
        # animate=['mass3D']
        # animate=['decoded-fixed']
        # animate=['latent3D', 'latent2D', 'data3D', 'data1D'],
        # plot=['latent3D', 'latent2D', 'data3D']
        # plot=['decoded']

        # final
        # t_plot=-1, final=True, plot=['latent3D-all'],
        # t_plot=11, final=True, plot=['data3D'],
        # t_plot=23, final=True, plot=['data3D'],
        # t_plot=35, final=True, plot=['data3D'],
        # t_plot=35, final=True, plot=['decoded3D-fixed-pre'],

        # final - mom
        # t_plot=-1, final=True, plot=['latent3D-all'], savepresuf=('mom_', ''), latent_type='mom',
        t_plot=-1, final=True, plot=['latent2D'], savepresuf=('mom_', ''), latent_type='mom',
        # t_plot=11, final=True, plot=['data3D'],
        # t_plot=23, final=True, plot=['data3D'],
        # t_plot=35, final=True, plot=['data3D'],
        # t_plot=35, final=True, plot=['decoded3D-fixed-pre'],

    )

    # -------------------
    # Final paper figures
    # -------------------

    # render = ['1D', '3D']
    render = ['']
    lt = 'mom'
    # lt = 'vae'
    final = True
    # pre = 'tmp_'
    pre = 'vae_' if lt == 'vae' else 'mom_'

    if '3D' in render:
        print('3D')
        kwargs = dict(model=latent_encoder, decoder=latent_decoder, final=final, plot=['data3D'], latent_type=lt)
        data_plot = Data(DATA_ATEX_ALL[0]).dataloader(batch_size=100000, by_time=True, t_plot=[11, 23, 35, 41])
        plot_latent_space(data_plot, **kwargs, t_plot=11, savepresuf=(pre, '_2x1'))
        plot_latent_space(data_plot, **kwargs, t_plot=23, savepresuf=(pre, '_4x1'))
        plot_latent_space(data_plot, **kwargs, t_plot=35, savepresuf=(pre, '_6x1'))
        plot_latent_space(data_plot, **kwargs, t_plot=41, savepresuf=(pre, '_7x1'))

        # data_plot = Data(DATA_ATEX_ALL[1]).dataloader(batch_size=100000, by_time=True, t_plot=[11, 23, 35, 41])
        # plot_latent_space(data_plot, **kwargs, t_plot=11, savepresuf=(pre, '_2x05'))
        # plot_latent_space(data_plot, **kwargs, t_plot=23, savepresuf=(pre, '_4x05'))
        # plot_latent_space(data_plot, **kwargs, t_plot=35, savepresuf=(pre, '_6x05'))
        # plot_latent_space(data_plot, **kwargs, t_plot=41, savepresuf=(pre, '_7x05'))

        # data_plot = Data(DATA_ATEX_ALL[2]).dataloader(batch_size=100000, by_time=True, t_plot=[11, 23, 35, 41])
        # plot_latent_space(data_plot, **kwargs, t_plot=11, savepresuf=(pre, '_2x2'))
        # plot_latent_space(data_plot, **kwargs, t_plot=23, savepresuf=(pre, '_4x2'))
        # plot_latent_space(data_plot, **kwargs, t_plot=35, savepresuf=(pre, '_6x2'))
        # plot_latent_space(data_plot, **kwargs, t_plot=41, savepresuf=(pre, '_7x2'))

    if '1D' in render:
        print('1D')
        kwargs = dict(model=latent_encoder, decoder=latent_decoder, final=final, plot=['data1D'])
        data_plot = Data(DATA_ATEX_ALL[0]).dataloader(batch_size=100000, by_time=True, t_plot=[11, 23, 35, 41])
        plot_latent_space(data_plot, **kwargs, t_plot=11, savepresuf=(pre, '_2x1_small'))
        plot_latent_space(data_plot, **kwargs, t_plot=23, savepresuf=(pre, '_4x1_small'))
        # plot_latent_space(data_plot, **kwargs, t_plot=35, savepresuf=(pre, '_6x1_small'))
        plot_latent_space(data_plot, **kwargs, t_plot=41, savepresuf=(pre, '_7x1_small'))

        data_plot = Data(DATA_ATEX_ALL[1]).dataloader(batch_size=100000, by_time=True, t_plot=[11, 23, 35, 41])
        plot_latent_space(data_plot, **kwargs, t_plot=11, savepresuf=(pre, '_2x05_small'))
        plot_latent_space(data_plot, **kwargs, t_plot=23, savepresuf=(pre, '_4x05_small'))
        # plot_latent_space(data_plot, **kwargs, t_plot=35, savepresuf=(pre, '_6x05_small'))
        plot_latent_space(data_plot, **kwargs, t_plot=41, savepresuf=(pre, '_7x05_small'))

        data_plot = Data(DATA_ATEX_ALL[2]).dataloader(batch_size=100000, by_time=True, t_plot=[11, 23, 35, 41])
        plot_latent_space(data_plot, **kwargs, t_plot=11, savepresuf=(pre, '_2x2_small'))
        plot_latent_space(data_plot, **kwargs, t_plot=23, savepresuf=(pre, '_4x2_small'))
        # plot_latent_space(data_plot, **kwargs, t_plot=35, savepresuf=(pre, '_6x2_small'))
        plot_latent_space(data_plot, **kwargs, t_plot=41, savepresuf=(pre, '_7x2_small'))
