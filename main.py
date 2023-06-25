from visualization import plot_latent_space, plot_components
from gmm_torch import latent_cluster
from data import DATA_ATEX_ALL, DATA_DYCOMS_ALL, Data

if __name__ == '__main__':
    # Data
    # data_subset = Data(os.path.splitext(DATA_PATH)[0] + '_nz.data').dataloader(batch_size=4096, num_workers=4)
    data = Data(DATA_ATEX_ALL).dataloader(batch_size=25000, weak_shuffle=True, num_workers=4)
    data_plot = Data(DATA_ATEX_ALL[0]).dataloader(batch_size=100000, by_time=True)

    latent_encoder, latent_decoder, gmm = latent_cluster(load_path='models/atex_mse.cp')

    # plot_components(gmm, latent_decoder)
    plot_latent_space(data_plot, latent_encoder, decoder=latent_decoder, gmm=None, t_plot=25,
                      # plot=['latent3D', 'latent2D', 'data3D', 'data1D']
                      # plot=['decoded-fixed', 'latent3D-fixed', 'latent2D-fixed']
                      # animate=['decoded-fixed']
                      # animate=['latent3D', 'latent2D', 'data3D', 'data1D'],
                      # plot=['latent3D', 'latent2D', 'data3D']
                      plot=['decoded']
                      )
