import sys, os
import argparse
from tools import progress
from data import nc2hdf5, Data
from gmm_torch import latent_cluster
from visualization import plot_latent_space

"""
Executable to run on cluster
"""


if __name__ == "__main__":
    """
    input:
    argv[1] - [data, train, animate] job type
    
    data:
    --src - source folder(s) for raw data
    --tgt - target folder(s) for preprocessed data
    
    train:
    argv[2]       - target folder(s) for preprocessed data
    --save        - path to save parameters checkpoint to
    --load        - path to load parameters checkpoint from
    --loss        - [mse, cor, max] reconstruction loss type, default=mse
    --latent_dims - dimensions of the latent space, default=3
    --hidden_dims - size of the hidden layers in the encoder/decoder, default=1024
    --beta        - VAE loss weight for the KL loss, default=0.001
    --epochs      - number of epochs to train, default=1
    
    animate:
    argv[2]   - target folder for preprocessed data
    --animate - [latent3D, latent3D-cluster, latent2D, latent2D-cluster, data3D, data3D-cluster, data1D, decoded-fixed]
                animation task(s), see visualization.py
    --plot    - plot task(s)
    --t_plot  - time step for plot tasks, default=25
    --load        - path to load parameters checkpoint from
    --latent_dims - dimensions of the latent space, default=3
    --hidden_dims - size of the hidden layers in the encoder/decoder, default=1024
    """

    kind = sys.argv.pop(1)
    if kind == 'data':
        parser = argparse.ArgumentParser(description='Preprocess data')
        parser.add_argument('--src', nargs='+', help='source folder(s) for raw data')
        parser.add_argument('--tgt', nargs='+', help='target folder(s) for preprocessed data')
        args = parser.parse_args()

        for src_dir, tgt_dir in zip(args.src, args.tgt):
            for f in progress(os.listdir(src_dir), 'Preprocessing'):
                nc2hdf5(os.path.join(src_dir, f), os.path.join(tgt_dir, os.path.splitext(f)[0] + '_nz.data'))

    elif kind == 'train':
        # todo: multi-folder
        parser = argparse.ArgumentParser(description='Train latent embedding')
        parser.add_argument('data_folder', nargs='+', help='target folder(s) for preprocessed data')
        parser.add_argument('--save', dest='save_path', default=None, help='path to save parameters checkpoint to')
        parser.add_argument('--load', dest='load_path', default=None, help='path to load parameters checkpoint from')
        parser.add_argument('--loss', dest='loss_type', default='mse',
                            help='reconstruction loss type, from [mse, cor, max]')
        parser.add_argument('--epochs', default=1, type=int, help='number of epochs to train')
        parser.add_argument('--latent_dims', default=3, type=int, help='dimensions of the latent space')
        parser.add_argument('--hidden_dims', default=1024, type=int,
                            help='size of the hidden layers in the encoder/decoder')
        parser.add_argument('--beta', default=0.001, type=float, help='VAE loss weight for the KL loss')
        kwargs = vars(parser.parse_args())

        data = Data(kwargs.pop('data_folder')).dataloader(batch_size=25000, weak_shuffle=True, num_workers=4)
        latent_cluster(**vars(parser.parse_args()))
    elif kind == 'animate':
        parser = argparse.ArgumentParser(description='Visualize latent embedding')
        parser.add_argument('data_folder', help='target folder for preprocessed data')
        parser.add_argument('--animate', nargs='*', help='animation task(s)')
        parser.add_argument('--plot', nargs='*', help='plot task(s)')
        parser.add_argument('--t_plot', default=25, type=int, help='time step for plot tasks')
        parser.add_argument('--load', dest='load_path', required=True, help='path to load parameters checkpoint from')
        parser.add_argument('--latent_dims', default=3, type=int, help='dimensions of the latent space')
        parser.add_argument('--hidden_dims', default=1024, type=int,
                            help='size of the hidden layers in the encoder/decoder')
        kwargs = vars(parser.parse_args())
        load_kwargs = {k: v for k, v in kwargs.items() if k in ['load_path', 'latent_dims', 'hidden_dims']}
        plot_kwargs = {k: v for k, v in kwargs.items() if k in ['animate', 'plot', 't_plot']}

        data = Data(kwargs['data_folder']).dataloader(batch_size=100000, by_time=True)
        latent_encoder, latent_decoder, _ = latent_cluster(**load_kwargs)
        plot_latent_space(data, latent_encoder, decoder=latent_decoder, **plot_kwargs)
    else:
        ValueError('Unknown task')
