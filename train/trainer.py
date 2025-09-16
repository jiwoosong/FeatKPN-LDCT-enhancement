import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path'       , type=str, default=r'../Data')
    parser.add_argument('--subject_path'    , type=str, default=r'../Data/train_val_split.pickle')
    parser.add_argument('--full_data'       , type=bool, default=False)

    parser.add_argument('--pin_memory'      , type=bool, default=True)
    parser.add_argument('--batch_size'      , type=int, default=1)
    parser.add_argument('--num_workers'     , type=int, default=4)
    parser.add_argument('--load_memory'     , type=bool, default=True)
    parser.add_argument('--window'          , type=int, default=[(-1024, 3072)])

    parser.add_argument('--model_name'      , type=str, default='KPNFeat')
    parser.add_argument('--patch_n'         , type=int)
    parser.add_argument('--patch_size'      , type=int)
    parser.add_argument('--nf'              , type=int, default=16)
    parser.add_argument('--nk'              , type=int, default=16)

    parser.add_argument('--ana_model_path'  , type=str, default=r'../Model/Anatomical_Segmentation/model.pth')
    parser.add_argument('--loss_type'       , type=str, default='l1_l1ana_1')
    parser.add_argument('--ana_weight'      , type=float, default=0.01)
    parser.add_argument('--lr'              , type=float, default=1e-4)
    parser.add_argument('--lrdecay'         , type=float, default=0.99)
    parser.add_argument('--pytorch_init'    , type=bool, default=False)


    parser.add_argument('--num_epochs'      , type=int, default=400)
    # parser.add_argument('--num_gpus'        , type=int, default=1) # Deprecated
    # parser.add_argument('--swa'             , type=bool, default=True) # Deprecated
    parser.add_argument('--precision'       , type=str, default=16)

    args = parser.parse_args()
    model_name = '%s_nf%d_nk%d'%(args.model_name, args.nf, args.nk) if args.model_name == 'KPNFeat' else args.model_name
    args.experiment_name = os.path.join(r'../../res/', '%s_(Full_%s)(L_%s)(B_%d)(%s)(lr_%.0e)_lambda'
                                        % (model_name,
                                           args.full_data,
                                           args.loss_type,
                                           args.batch_size,
                                           str(args.window),
                                           args.lr
                                           ))

    if not os.path.isdir(args.experiment_name):os.makedirs(args.experiment_name)
    tb_logger = TensorBoardLogger(save_dir=args.experiment_name)
    checkpointing = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.experiment_name, f'version_{tb_logger.version}'),
        filename='E{epoch:2d}_MSE{val/MSE:.3f}_AVGPSNR{val/Avg_PSNR:.2f}_AVGSSIM{val/Avg_SSIM:.4f}',
        save_last=False,
        save_top_k=5,
        monitor='val/MSE',
        mode='min',
        auto_insert_metric_name=False
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(max_epochs = args.num_epochs,
                         # gpus = args.num_gpus,
                         callbacks = [checkpointing, lr_monitor],
                         logger = tb_logger,
                         precision = args.precision,
                         # stochastic_weight_avg = args.swa,
                         # accelerator="ddp",
                         num_sanity_val_steps = 0,
                         # plugins=DDPPlugin(find_unused_parameters=False),
                         )


    from loader import get_Full_Denoising_Loader
    train_loader, val_loader = get_Full_Denoising_Loader(args)

    # Model
    from Model.solver import ADE_solver
    model = ADE_solver(args)

    # Start Training!!
    trainer.fit(model, train_loader, val_loader)