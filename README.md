# Data Analysis for Droplet Distributions
Understanding droplet distribution simulation through latent spaces analysis.
Run preprocessing, training and visualization via `run.py`:

```PowerShell
python run.py data -h
```
```
> usage: run.py [-h] [--src SRC [SRC ...]] [--tgt TGT [TGT ...]]
> 
> Preprocess data
> 
> optional arguments:
>   -h, --help           show this help message and exit
>   --src SRC [SRC ...]  source folder(s) for raw data
>   --tgt TGT [TGT ...]  target folder(s) for preprocessed data
```

```PowerShell
python run.py train 'data_folder' -h
```
```
> usage: run.py [-h] [--save SAVE_PATH] [--load LOAD_PATH] [--loss LOSS_TYPE] [--latent_dims LATENT_DIMS] [--hidden_dims HIDDEN_DIMS] [--beta BETA]
              data_folder [data_folder ...]
> 
> Train latent embedding
> 
> positional arguments:
>   data_folder           target folder(s) for preprocessed data
> 
> optional arguments:
>   -h, --help            show this help message and exit
>   --save SAVE_PATH      path to save parameters checkpoint to
>   --load LOAD_PATH      path to load parameters checkpoint from
>   --loss LOSS_TYPE      reconstruction loss type, from [mse, cor, max]
>   --latent_dims LATENT_DIMS
>                         dimensions of the latent space
>   --hidden_dims HIDDEN_DIMS
>                         size of the hidden layers in the encoder/decoder
>   --beta BETA           VAE loss weight for the KL loss
```

```PowerShell
python run.py animate 'data_folder' -h
```
```
> usage: run.py [-h] [--animate [ANIMATE ...]] [--plot [PLOT ...]] [--t_plot T_PLOT] --load LOAD_PATH [--latent_dims LATENT_DIMS] [--hidden_dims HIDDEN_DIMS]
>               data_folder
> 
> Visualize latent embedding
> 
> positional arguments:
>   data_folder           target folder for preprocessed data
> 
> optional arguments:
>   -h, --help            show this help message and exit
>   --animate [ANIMATE ...]
>                         animation task(s)
>   --plot [PLOT ...]     plot task(s)
>   --t_plot T_PLOT       time step for plot tasks
>   --load LOAD_PATH      path to load parameters checkpoint from
>   --latent_dims LATENT_DIMS
>                         dimensions of the latent space
>   --hidden_dims HIDDEN_DIMS
>                         size of the hidden layers in the encoder/decoder
```
