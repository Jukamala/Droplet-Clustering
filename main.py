from visualization import plot_latent_space, plot_components
from gmm_torch import latent_cluster
from data import DATA_ATEX_ALL, DATA_DYCOMS_ALL, Data

if __name__ == '__main__':
    # Data
    data = Data(DATA_ATEX_ALL).dataloader(batch_size=25000, weak_shuffle=True, num_workers=4)
    # data_plot = Data(DATA_ATEX_ALL[0]).dataloader(batch_size=100000, by_time=True, t_plot=25)
    data_plot = Data(DATA_ATEX_ALL[0]).dataloader(batch_size=100000, by_time=True)
    # data_plot = Data(DATA_ATEX_ALL).dataloader(batch_size=100000, by_time=True)

    # Train via: latent_cluster(data, load_path='models/atex_mse.cp')
    latent_encoder, latent_decoder, gmm = latent_cluster(load_path='models/atex_mse.cp')

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
                      # t_plot=11, final=True, plot=['data1D'],
                      # t_plot=23, final=True, plot=['data1D'],
                      t_plot=35, final=True, plot=['data1D'],
                      )
