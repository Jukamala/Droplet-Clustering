from visualization import plot_latent_space, plot_components
from gmm_torch import latent_cluster
from data import DATA_FOLDER, Data


if __name__ == '__main__':
    # Data
    # data_subset = Data(os.path.splitext(DATA_PATH)[0] + '_nz.data').dataloader(batch_size=4096, num_workers=4)
    data = Data(DATA_FOLDER).dataloader(batch_size=25000, weak_shuffle=True, num_workers=4)
    data_plot = Data(DATA_FOLDER).dataloader(batch_size=100000, by_time=True)

    latent_encoder, latent_decoder, gmm = latent_cluster(data)

    # plot_components(gmm, latent_decoder)
    plot_latent_space(data_plot, latent_encoder, decoder=latent_decoder, gmm=None, t_plot=25,
                      # plot=['decoded-fixed', 'latent3D-fixed', 'latent2D-fixed']
                      # plot = ['latent3D', 'latent2D', 'data3D', 'data1D']
                      animate=['decoded-fixed']
                      # plot=['decoded-fixed']
                      )
    # plot_latent_space(data_plot, latent_encoder, gmm=None, t_plot=25,
    #                   plot=['latent3D', 'latent2D', 'data3D'], animate=['latent3D', 'latent2D', 'data3D', 'data1D'])
