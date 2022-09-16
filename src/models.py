import torch
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD, Adam
from src.helper import *
from src.unet import *
from src.metrics import *
from src.unet_quadconv import *
import pdb
import statsmodels.stats.api as sms


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

# Lightning Module class
class LitSegmentation(pl.LightningModule):
    def __init__(self, args, model_name):
        super().__init__()
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers['adam']
        self.save_hyperparameters(ignore=['model_name'])
        self.metric1 = dice_coef_multilabel
        self.model = model_name
        self.all_val_metrics = False #fix this
        # if self.all_val_metrics:
        self.metric2 = signed_thickness_diff_multilabel
        self.metric3 = unsigned_thickness_diff_multilabel
        self.metric4 = dice_coef_separate_multilabel
            

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        return {'loss':loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('step', torch.tensor(self.current_epoch, dtype=torch.float32))
        self.log("train_loss_epoch", avg_loss, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.binary_cross_entropy_with_logits(y_hat, y)

        y_hat_sig = torch.sigmoid(y_hat)
        y_hat_sig_max = torch.argmax(y_hat_sig, dim=1)
        y_hat_sig_max = torch.zeros_like(y_hat_sig).scatter_(1, y_hat_sig_max.unsqueeze(1), 1.)
        y = y.cpu().detach().numpy()
        y_hat_sig_max = y_hat_sig_max.cpu().detach().numpy()
        y_hat_sig = y_hat_sig.cpu().numpy()
        
        if self.hparams.args.all_val_metrics:       
            return {'val_loss':val_loss, 
            'dice_max':self.metric1(y, y_hat_sig_max, 8), 
            'dice':self.metric1(y, y_hat_sig, 8),
            'signed_diff': self.metric2(y, y_hat_sig_max, 8),
            'unsigned_diff': self.metric3(y, y_hat_sig_max, 8),
            'dice_coeff': self.metric4(y, y_hat_sig_max, 8)
            }
        else:
            return {'val_loss':val_loss,
            'dice_max':self.metric1(y, y_hat_sig_max, 8), 
            'dice':self.metric1(y, y_hat_sig, 8)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_dice = np.stack([output['dice'][:,0] for output in outputs]).mean()
        avg_dice_max = np.stack([output['dice_max'][:,0] for output in outputs]).mean()
        
        self.log('step', torch.tensor(self.current_epoch, dtype=torch.float32))
        self.log("validation_loss", avg_loss, on_step=False, on_epoch=True)
        self.log('validation_dice', avg_dice, on_step=False, on_epoch=True)
        self.log('validation_dice_argmax', avg_dice_max, on_step=False, on_epoch=True)
        
        if self.hparams.args.all_val_metrics:
            avg_dice_max_std = np.stack([output['dice_max'][:,0] for output in outputs]).std()
            self.log('avg_dice_max_std', avg_dice_max_std, on_step=False, on_epoch=True)

            sd_name = ['signed_diff_%s' % i for i in range(1,8)]
            signed_diff_dict = {}
            for diff_i, name in enumerate(sd_name):
                data = np.stack([output['signed_diff'][diff_i + 1] for output in outputs]).flatten()
                signed_diff_dict[name] = data.mean()
                ci = sms.DescrStatsW(data).tconfint_mean()
                ci = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, ci))
                self.log(name, round(signed_diff_dict[name], 2), on_step = False, on_epoch=True)
                self.log(name+'_ci_l', ci[0], on_step = False, on_epoch=True)
                self.log(name+'_ci_u', ci[1], on_step = False, on_epoch=True)

            sd_name = ['unsigned_diff_%s' % i for i in range(1,8)]
            unsigned_diff_dict = {}
            for diff_i, name in enumerate(sd_name):
                data = np.stack([output['unsigned_diff'][diff_i + 1] for output in outputs]).flatten()
                unsigned_diff_dict[name] = data.mean()
                ci = sms.DescrStatsW(data).tconfint_mean()
                ci = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, ci))
                self.log(name, round(unsigned_diff_dict[name], 2), on_step = False, on_epoch=True)
                self.log(name+'_ci_l', ci[0], on_step = False, on_epoch=True)
                self.log(name+'_ci_u', ci[1], on_step = False, on_epoch=True)

            sd_name = ['dice_coeff_%s' % i for i in range(1,8)]
            dice_coeff_dict = {}
            for diff_i, name in enumerate(sd_name):
                data = np.stack([output['dice_coeff'][diff_i + 1] for output in outputs]).flatten()
                dice_coeff_dict[name] = data.mean()
                ci = sms.DescrStatsW(data).tconfint_mean()
                ci = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, ci))
                self.log(name, round(dice_coeff_dict[name], 2), on_step = False, on_epoch=True)
                self.log(name+'_ci_l', ci[0], on_step = False, on_epoch=True)
                self.log(name+'_ci_u', ci[1], on_step = False, on_epoch=True)
                         
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        test_loss = F.binary_cross_entropy_with_logits(y_hat, y)

        y_hat_sig = torch.sigmoid(y_hat)
        y_hat_sig_max = torch.argmax(y_hat_sig, dim=1)
        y_hat_sig_max = torch.zeros_like(y_hat_sig).scatter_(1, y_hat_sig_max.unsqueeze(1), 1.)
        y = y.cpu().detach().numpy()
        y_hat_sig_max = y_hat_sig_max.cpu().detach().numpy()
        y_hat_sig = y_hat_sig.cpu().numpy()
        
        
        return {'test_loss':test_loss, 
        'dice_max':self.metric1(y, y_hat_sig_max, 8), 
        'dice':self.metric1(y, y_hat_sig, 8),
        'signed_diff': self.metric2(y, y_hat_sig_max, 8),
        'unsigned_diff': self.metric3(y, y_hat_sig_max, 8),
        }
        
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_dice = np.concatenate([output['dice'][:,0] for output in outputs]).mean()
        avg_dice_max = np.concatenate([output['dice_max'][:,0] for output in outputs]).mean()
        
        self.log('step', torch.tensor(self.current_epoch, dtype=torch.float32))
        self.log("test_loss", avg_loss, on_step=False, on_epoch=True)
        self.log('test_dice', avg_dice, on_step=False, on_epoch=True)
        self.log('test_dice_argmax', avg_dice_max, on_step=False, on_epoch=True)
        
        avg_dice_max_std = np.concatenate([output['dice_max'][:,0] for output in outputs]).std()
        signed_overall_diff = np.concatenate([output['signed_diff'][:,0] for output in outputs]).flatten().mean() * (2000/1024)
        signed_overall_diff_std = np.concatenate([output['signed_diff'][:,0] for output in outputs]).flatten().std() * (2000/1024)
        unsigned_overall_diff = np.concatenate([output['unsigned_diff'][:,0] for output in outputs]).flatten().mean() * (2000/1024)
        unsigned_overall_diff_std = np.concatenate([output['unsigned_diff'][:,0] for output in outputs]).flatten().std() * (2000/1024)
            
        self.log('avg_dice_max_std', avg_dice_max_std, on_step=False, on_epoch=True)
        self.log("signed_average_difference", signed_overall_diff, on_step=False, on_epoch=True)
        self.log("signed_average_difference_std", signed_overall_diff_std, on_step=False, on_epoch=True)
        self.log("unsigned_average_difference", unsigned_overall_diff, on_step=False, on_epoch=True)
        self.log("unsigned_average_difference_std", unsigned_overall_diff_std, on_step=False, on_epoch=True)
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        # pdb.set_trace()
        y_hat_sig = torch.sigmoid(y_hat)
        y_hat_sig_max = torch.argmax(y_hat_sig, dim=1)
        y_hat_sig_max = torch.zeros_like(y_hat_sig).scatter_(1, y_hat_sig_max.unsqueeze(1), 1.)
         
        return y_hat_sig_max
         
    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), lr=self.hparams.args.learning_rate)
        scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
        return [optimizer], [scheduler]