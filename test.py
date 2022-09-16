import argparse
import pytorch_lightning as pl
import sys
from src.helper import *
from src.models import *
from src.dataset import SegmentationDataModule

def cli_main():
    parser = argparse.ArgumentParser()
    
    #important values to change
    parser.add_argument("--model_name", type=str, default = "UnetL4_skip")#, type=str, choices=['ResnetUNet23'])
    parser.add_argument("--ckpt_path", type=str, default = "lightning_logs/UnetL4_skip/version_0/checkpoints/unet-epoch=99-validation_dice_argmax=0.91.ckpt")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--all_val_metrics", default=True)
    #args
    parser.add_argument("--path_csv", default='dataset/dataset.csv')
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--version", type=str, default=None)
    #Find the optimal value for these then leave as default values
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--calculate_num_workers", default=False, help="set as true if you are using a new machine")    
    #mean and standard deviation changes with chosen global seed
    parser.add_argument("--mean", type=float, default=0.1493, help="calculated from previous run")
    parser.add_argument("--std", type=float, default=0.1001, help="calculated from previous run")
    parser.add_argument("--calculate_mean_std", default=False, help="set as true if using different seed")
    #print model parameters
    parser.add_argument("--model_parameters", default = False)
    parser.add_argument("--validate_all", default=False)

    args = parser.parse_args()
    
    #set random seed
    pl.seed_everything(args.random_seed)
    path_save = 'dataset/random_seed_'+str(args.random_seed)+'.csv'

    #model options
    models = {'UnetL3': UNet2D(1, 8, conv_depths=(64, 128, 256), batch_norm=True),
            'UnetL3_skip': UNet2D(1, 8, conv_depths=(64, 128, 256), skip=True, batch_norm=True),
            'UnetL3_attention': UNet2D(1, 8, conv_depths=(64, 128, 256), attention=True, batch_norm=True),
            'UnetL3_skip_attention': UNet2D(1, 8, conv_depths=(64, 128, 256), skip=True, attention=True, batch_norm=True),
            'UnetL3_quadconv': UNet2D_quadconv(1, 8, conv_depths=(64, 128, 256), batch_norm=True),
            'UnetL3_quadconv_skip': UNet2D_quadconv(1, 8, conv_depths=(64, 128, 256), skip=True, batch_norm=True),
            'UnetL3_quadconv_attention': UNet2D_quadconv(1, 8, conv_depths=(64, 128, 256), attention=True, batch_norm=True),
            'UnetL3_quadconv_skip_attention': UNet2D_quadconv(1, 8, conv_depths=(64, 128, 256), skip=True, attention=True, batch_norm=True),
            'UnetL4': UNet2D(1, 8, conv_depths=(64, 128, 256, 512), batch_norm=True),
            'UnetL4_skip': UNet2D(1, 8, conv_depths=(64, 128, 256, 512), skip=True, batch_norm=True),
            'UnetL4_attention': UNet2D(1, 8, conv_depths=(64, 128, 256, 512), attention=True, batch_norm=True),
            'UnetL4_skip_attention': UNet2D(1, 8, conv_depths=(64, 128, 256, 512), skip=True, attention=True, batch_norm=True),
            'UnetL4_quadconv': UNet2D_quadconv(1, 8, conv_depths=(64, 128, 256, 512), batch_norm=True),
            'UnetL4_quadconv_skip': UNet2D_quadconv(1, 8, conv_depths=(64, 128, 256, 512), skip=True, batch_norm=True),
            'UnetL4_quadconv_attention': UNet2D_quadconv(1, 8, conv_depths=(64, 128, 256, 512), attention=True, batch_norm=True),
            'UnetL4_quadconv_skip_attention': UNet2D_quadconv(1, 8, conv_depths=(64, 128, 256, 512), skip=True, attention=True, batch_norm=True),
            'UnetL5': UNet2D(1, 8, conv_depths=(64, 128, 256, 512, 1024), batch_norm=True),
            'UnetL5_skip': UNet2D(1, 8, conv_depths=(64, 128, 256, 512, 1024), skip=True, batch_norm=True),
            'UnetL5_attention': UNet2D(1, 8, conv_depths=(64, 128, 256, 512, 1024), attention=True, batch_norm=True),
            'UnetL5_skip_attention': UNet2D(1, 8, conv_depths=(64, 128, 256, 512, 1024), skip=True, attention=True, batch_norm=True),
            'UnetL5_quadconv': UNet2D_quadconv(1, 8, conv_depths=(64, 128, 256, 512, 1024), batch_norm=True),
            'UnetL5_quadconv_skip': UNet2D_quadconv(1, 8, conv_depths=(64, 128, 256, 512, 1024), skip=True, batch_norm=True),
            'UnetL5_quadconv_attention': UNet2D_quadconv(1, 8, conv_depths=(64, 128, 256, 512, 1024), attention=True, batch_norm=True),
            'UnetL5_quadconv_skip_attention': UNet2D_quadconv(1, 8, conv_depths=(64, 128, 256, 512, 1024), skip=True, attention=True, batch_norm=True),}
    model_name = models[args.model_name]

    if args.validate_all:
        from pathlib import Path
        import json
        diff = {}
        for key, value in sorted(models.items()):
            files =  [path for path in Path('lightning_logs/').rglob(key+'/*/checkpoints/*.ckpt')]
            files.sort()
            args.ckpt_path = files[-1]
            model_name = value
            model = LitSegmentation(args, model_name)
            segmentation_dataloader = SegmentationDataModule(args.mean, args.std, args.path_csv, path_save, args.batch_size, args.num_workers)
            trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
            results = trainer.validate(model, datamodule=segmentation_dataloader, ckpt_path = args.ckpt_path)
            # diff[key] = {'validation_dice_argmax':results[0]['validation_dice_argmax'], 'average_difference':results[0]['average_difference'], 'average_difference_unsigned':results[0]['average_difference_unsigned']}
            diff[key] = results
        print(diff)
        with open('validation_results_single.json', 'w') as f:
            json.dump(diff, f)
        sys.exit()
    
    #initiate model LitSegmentation class
    model = LitSegmentation(args, model_name)
    segmentation_dataloader = SegmentationDataModule(args.mean, args.std, args.path_csv, path_save, args.batch_size, args.num_workers)
    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
    results = trainer.test(model, datamodule=segmentation_dataloader, ckpt_path = args.ckpt_path)
    print(results)
if __name__ == '__main__':
    cli_main()