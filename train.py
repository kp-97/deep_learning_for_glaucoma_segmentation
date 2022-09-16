import argparse
import pytorch_lightning as pl
import sys
from torchvision import transforms
from torch.utils.data import DataLoader#, Dataset,
from torchvision import transforms
from src.helper import *
from src.models import *
from src.dataset import SegmentationDataModule, SegmentationDataset
from pytorch_lightning.loggers import TensorBoardLogger

def cli_main():
    parser = argparse.ArgumentParser()
    
    #important values to change
    parser.add_argument("--model_name", type=str, default = "UnetL3_attention")#, type=str, choices=['ResnetUNet23'])
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="initial learning rate for scheduler")
    parser.add_argument("--max_epochs", type=int, default=100)
    #args
    parser.add_argument("--path_csv", default='dataset/dataset.csv')
    parser.add_argument("--ckpt_path", type=str, help="provide relative path to the .ckpt path to continue training")
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--version", type=str, default=None)
    #Find the optimal value for these then leave as default values, batch size is found automatically
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--calculate_num_workers", default=False, help="set as true if you are using a new machine")    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--find_batchsize", default=True)
    #mean and standard deviation changes with chosen global seed
    parser.add_argument("--mean", type=float, default=0.1493, help="calculated from previous run")
    parser.add_argument("--std", type=float, default=0.1001, help="calculated from previous run")
    parser.add_argument("--calculate_mean_std", default=False, help="set as true if using different seed")
    #print model parameters
    parser.add_argument("--model_parameters", default = False)
    #debug
    parser.add_argument("--debug", default=False)
    
    args = parser.parse_args()
    
    #set random seed
    pl.seed_everything(args.random_seed)
    path_save = 'dataset/random_seed_'+str(args.random_seed)+'.csv'

    #calculate mean and standard deviation if a new seed is used
    if args.calculate_mean_std:
        trans = transforms.Compose([
        transforms.ToTensor()
        ])
        train_set = SegmentationDataset(args.path_csv, path_save, 'train', transform = trans)
        train_loader = DataLoader(train_set, batch_size= args.batch_size, num_workers=args.num_workers)    
        Calculations.calc_mean_std(train_loader)
        sys.exit()

    #image specific transforms    
    trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((args.mean), (args.std))
    ])

    #mask specific transforms
    mask_trans = transforms.Compose([
    transforms.ToTensor()
    ])

    #calculate num_workers, use this if you are on a new machine
    if args.calculate_num_workers:
        train_set = SegmentationDataset(args.path_csv, path_save, 'train', transform = trans, mask_transform = mask_trans)
        Calculations.calc_num_workers(train_set, args.batch_size)
        sys.exit()
    
    #model options, look at src/models.py
    model_name = models[args.model_name]
    
    #print model parameters
    if args.model_parameters:
        from pytorch_lightning.utilities.model_summary import ModelSummary
        model = LitSegmentation(args, model_name)
        print(ModelSummary(model, max_depth=0))
        sys.exit()

    #find batch_size using pl, doesnt work properly, bin search results in ineligible batch size
    # trainer = pl.Trainer(auto_scale_batch_size=True, accelerator="gpu", devices=1)
    # trainer.tune(model, datamodule=segmentation_dataloader)

    #use this for debugging, 
    if args.debug:
        batch_size = 1
        model = LitSegmentation(args, model_name)
        segmentation_dataloader = SegmentationDataModule(args.mean, args.std, args.path_csv, path_save, batch_size, args.num_workers)
        trainer = pl.Trainer(fast_dev_run=3, accelerator="gpu", devices=1, enable_model_summary=False, enable_progress_bar=False, weights_summary=False)
        trainer.fit(model, datamodule=segmentation_dataloader)
        print('successful run')
        sys.exit()

    #find batch size (custom)
    if args.find_batchsize:
        for batch_size in range(1,100):
            try:
                model = LitSegmentation(args, model_name)
                segmentation_dataloader = SegmentationDataModule(args.mean, args.std, args.path_csv, path_save, batch_size, args.num_workers)
                trainer = pl.Trainer(fast_dev_run=3, accelerator="gpu", devices=1, enable_model_summary=False, enable_progress_bar=False, weights_summary=False)
                trainer.fit(model, datamodule=segmentation_dataloader)
                args.batch_size = batch_size
            except:
                break

    #initiate model LitSegmentation class
    model = LitSegmentation(args, model_name)
    segmentation_dataloader = SegmentationDataModule(args.mean, args.std, args.path_csv, path_save, args.batch_size, args.num_workers)
    
    if args.ckpt_path:
        logger = TensorBoardLogger("lightning_logs", version = args.ckpt_path.split('/')[-3], name=args.model_name)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=True, filename="unet-{epoch}-{validation_dice_argmax:.2f}", save_top_k=3, monitor="validation_dice_argmax", mode="max")
        trainer = pl.Trainer(callbacks=[checkpoint_callback], accelerator="gpu", devices=1, log_every_n_steps=40, max_epochs=args.max_epochs, logger=logger)
        trainer.fit(model, datamodule=segmentation_dataloader, ckpt_path = args.ckpt_path)

    else:
        logger = TensorBoardLogger("lightning_logs", version = args.version, name=args.model_name)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=True, filename="unet-{epoch}-{validation_dice_argmax:.2f}", save_top_k=3, monitor="validation_dice_argmax", mode="max")
        trainer = pl.Trainer(callbacks=[checkpoint_callback], accelerator="gpu", devices=1, log_every_n_steps=40, max_epochs=args.max_epochs, logger=logger)
        trainer.fit(model, datamodule=segmentation_dataloader)

if __name__ == '__main__':
    cli_main()