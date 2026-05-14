import torch
import pytorch_lightning as pl
import torch.nn as nn 
import torch.nn.functional as F
from contextlib import contextmanager
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from taming.modules.vqvae.quantize import VectorQuantizer as VectorQuantizer

from modules.diffusionmodules.model import Encoder, Decoder
from modules.distributions.distributions import DiagonalGaussianDistribution
from convlstm import ConvLSTMCell
from data_preprocessing import preprocess_batch

from util import instantiate_from_config

IMG_SIZE = 64
SEQ_LEN = 10

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_hiddens)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

# Encoder & Decoder Architecture:
# Encoder:
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=in_channels,
                                                  out_channels=num_hiddens//2,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens//2),
                                        nn.ReLU()
                                    ])
        self._conv_2 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=num_hiddens//2,
                                                  out_channels=num_hiddens,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens)
                                        #nn.ReLU()
                                    ])
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._conv_2(x)
        x = self._residual_stack(x)
        return x # predicted_map 

# Decoder:
class Decoder(nn.Module):
    def __init__(self, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_2 = nn.Sequential(*[
                                            nn.ReLU(),
                                            nn.ConvTranspose2d(in_channels=num_hiddens,
                                                              out_channels=num_hiddens//2,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1),
                                            nn.BatchNorm2d(num_hiddens//2),
                                            nn.ReLU()
                                        ])

        self._conv_trans_1 = nn.Sequential(*[
                                            nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                              out_channels=num_hiddens//2,
                                                              kernel_size=4,
                                                              stride=2,
                                                              padding=1),
                                            nn.BatchNorm2d(num_hiddens//2),
                                            nn.ReLU(),                  
                                            nn.Conv2d(in_channels=num_hiddens//2,
                                                      out_channels=out_channels,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1),
                                            nn.Sigmoid() # nn.Tanh()
                                        ])

    def forward(self, inputs):
        x = self._residual_stack(inputs)
        x = self._conv_trans_2(x)
        x = self._conv_trans_1(x)
        return x


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class SequenceAutoencoderKL(pl.LightningModule):
    """
    Sequence VAE:
        input  : (B, T, C, H, W)
        latent : (B, embed_dim, H_lat, W_lat)
        output : (B, T, C, H, W)

    Designed for PredOcc occupancy-map sequences.
    """

    def __init__(
        self,
        lossconfig,
        embed_dim,
        seq_len=20,
        in_channels=1,
        out_ch=1,
        resolution=64,
        temporal_hidden_dim=32,
        num_hiddens=128,
        num_residual_layers=2,
        num_residual_hiddens=64,
        image_key="image",
        monitor=None,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.image_key = image_key
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.temporal_hidden_dim = temporal_hidden_dim
        self.in_channels = in_channels
        self.out_ch = out_ch
        self.resolution = resolution
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens

        self._encoder = Encoder(
            in_channels=in_channels,
            num_hiddens=self.num_hiddens,
            num_residual_layers=self.num_residual_layers,
            num_residual_hiddens=self.num_residual_hiddens,
        )

        self._encoder_z_mu = nn.Conv2d(in_channels=num_hiddens, 
                                    out_channels=embed_dim,
                                    kernel_size=1, 
                                    stride=1)
        self._encoder_z_log_var = nn.Conv2d(in_channels=num_hiddens, 
                                    out_channels=embed_dim,
                                    kernel_size=1, 
                                    stride=1)  

        self._decoder_z_mu = nn.ConvTranspose2d(in_channels=embed_dim, 
                                    out_channels=num_hiddens,
                                    kernel_size=1, 
                                    stride=1)
        
        self._decoder = Decoder(
            out_channels= self.out_ch,
            num_hiddens=self.num_hiddens,
            num_residual_layers=self.num_residual_layers,
            num_residual_hiddens=self.num_residual_hiddens,
        )

        self.loss = instantiate_from_config(lossconfig)

        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]

        current_sd = self.state_dict()

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
                    break

        skipped = []
        for k in list(sd.keys()):
            if k not in current_sd:
                continue
            if sd[k].shape != current_sd[k].shape:
                skipped.append((k, tuple(sd[k].shape), tuple(current_sd[k].shape)))
                del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if skipped:
            print("Skipped mismatched keys:")
            for name, ckpt_shape, model_shape in skipped:
                print(f"  {name}: checkpoint {ckpt_shape} != model {model_shape}")
        if len(missing) > 0:
            print("Missing keys:", missing)
        if len(unexpected) > 0:
            print("Unexpected keys:", unexpected)

    # ------------------------------------------------------------------
    # Data handling
    # ------------------------------------------------------------------
    def get_input(self, batch, k):
        """
        Convert preprocessed maps to a 1-channel video tensor: (B, 2T, 1, H, W).
        First T frames: past (input), next T frames: future (target to predict).
        """
        maps = preprocess_batch(batch, device=self.device)

        input_binary_maps  = maps["input_binary_maps"].float()   # (B, T, 1, H, W) - past
        mask_binary_maps   = maps["mask_binary_maps"].float()    # (B, T, 1, H, W) - future

        x = torch.cat([input_binary_maps, mask_binary_maps], dim=1)  # (B, 2T, 1, H, W)

        return x

    # ------------------------------------------------------------------
    # Core VAE
    # ------------------------------------------------------------------
    def encode(self, x_seq, x_map=None):
        """
        x_seq: (B, T, C, H, W) = (B, 20, 1, 64, 64)
        returns: posterior over sequence of latents (B*T, embed_dim, 16, 16)
        
        Pipeline (frame-wise encoding):
        - Reshape: (B, T, 1, 64, 64) → (B*T, 1, 64, 64)
        - Encoder: (B*T, 1, 64, 64) → (B*T, 128, 16, 16)
        - z_mu/z_log_var: (B*T, embed_dim, 16, 16)
        """
        b, t, c, h, w = x_seq.shape
        
        # Step 1: Reshape to flatten time dimension
        x_flat = x_seq.reshape(b * t, c, h, w)  # (B*T, 1, 64, 64)
        
        # Step 2: Encoder
        feat = self._encoder(x_flat)  # (B*T, 128, 16, 16)
        
        # Step 3: z_mu, z_log_var (per-frame)
        z_mu = self._encoder_z_mu(feat)       # (B*T, embed_dim, 16, 16)
        z_log_var = self._encoder_z_log_var(feat)  # (B*T, embed_dim, 16, 16)
        
        # Step 4: Create moments for DiagonalGaussianDistribution
        moments = torch.cat([z_mu, z_log_var], dim=1)  # (B*T, 2*embed_dim, 16, 16)
        posterior = DiagonalGaussianDistribution(moments)
        
        return posterior

    def decode(self, z):
        """
        z: (B*T, embed_dim, 16, 16)
        returns: (B, T, C, H, W) = (B, 20, 1, 64, 64)
        
        Pipeline:
        - decoder_z_mu: (B*T, 2, 16, 16) → (B*T, 128, 16, 16)
        - Decoder: (B*T, 128, 16, 16) → (B*T, 1, 64, 64)
        - Reshape: (B*T, 1, 64, 64) → (B, T, 1, 64, 64)
        """
        b_t = z.shape[0]
        
        # Step 1: decoder_z_mu
        z = self._decoder_z_mu(z)  # (B*T, 128, 16, 16)
        
        # Step 2: Decoder
        dec = self._decoder(z)  # (B*T, 1, 64, 64)
        
        return dec

    def forward(self, input_seq, x_map=None, sample_posterior=False):
        """
        input_seq: (B, T, C, H, W)
        x_map: (unused in this pipeline)
        """
        posterior = self.encode(input_seq)
        z = posterior.sample() if sample_posterior else posterior.mode()  # (B*T, 2, 16, 16)
        dec = self.decode(z)
        return dec, posterior

    # ------------------------------------------------------------------
    # Training / validation
    # Manual optimization for modern Lightning multi-optimizer support
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)              # (B,2T,C,H,W)
        reconstructions, posterior = self(inputs)          # recon: (B,2T,C,H,W)

        opt_ae = self.optimizers()

        # Flatten time for the existing loss implementation
        b, t, c, h, w = inputs.shape

        inputs_2d = inputs.view(b * t, c, h, w)
        recon_2d = reconstructions.view(b * t, c, h, w)

        # 1) AE / generator update
        opt_ae.zero_grad()
        aeloss, log_dict_ae = self.loss(
            inputs_2d,
            recon_2d,
            posterior,
            split="train",
        )
        self.manual_backward(aeloss)
        opt_ae.step()

        self.log("train/total_loss", log_dict_ae["train/total_loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return aeloss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        b, t, c, h, w = inputs.shape

        inputs_2d = inputs.view(b * t, c, h, w)
        recon_2d = reconstructions.view(b * t, c, h, w)

        aeloss, log_dict_ae = self.loss(
            inputs_2d,
            recon_2d,
            posterior,
            split="val"
        )
        self.log("val/total_loss", log_dict_ae["val/total_loss"], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return log_dict_ae

    def configure_optimizers(self):
        lr = self.learning_rate

        opt_ae = torch.optim.Adam(
            list(self._encoder.parameters()) +
            list(self._decoder.parameters()) +
            list(self._encoder_z_mu.parameters()) +
            list(self._encoder_z_log_var.parameters()) +
            list(self._decoder_z_mu.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )

        return opt_ae
    
    def get_last_layer(self):
        return self._decoder._conv_trans_1[3].weight
    
    @torch.no_grad()
    def estimate_latent_stats(self, dataloader, num_batches=50, use_mode=False):
        """
        Estimate latent mean/std from encoded training sequences.

        Args:
            dataloader: PyTorch dataloader yielding PredOccDataset batches
            num_batches: how many batches to use
            use_mode: if True use posterior.mode(), else posterior.sample()

        Returns:
            dict with latent mean/std/min/max and recommended scale_factor
        """
        was_training = self.training
        self.eval()

        mean_list = []
        std_list = []
        min_list = []
        max_list = []

        for i, batch in enumerate(dataloader):
            if i >= num_batches: 
                break

            x = self.get_input(batch, self.image_key)
            x = x.to(self.device)  # (B,T,1,H,W)
            posterior = self.encode(x)
            z = posterior.mode() if use_mode else posterior.sample()    # (B,C_lat,H_lat,W_lat)

            mean_list.append(z.mean().item())
            std_list.append(z.std().item())
            min_list.append(z.min().item())
            max_list.append(z.max().item())

        stats = {
            "latent_mean": sum(mean_list) / len(mean_list),
            "latent_std": sum(std_list) / len(std_list),
            "latent_min": sum(min_list) / len(min_list),
            "latent_max": sum(max_list) / len(max_list),
        }
        stats["recommended_scale_factor"] = 1.0 / max(stats["latent_std"], 1e-8)

        print("\n[Latent Stats]")
        for k, v in stats.items():
            print(f"{k}: {v:.6f}")

        if was_training:
            self.train()

        return stats
    
    @staticmethod
    def compute_iou(pred, gt, occ_thr=0.3):
        pred_occ = (pred > occ_thr)
        gt_occ   = (gt > occ_thr)

        inter = (pred_occ & gt_occ).sum().float()
        union = (pred_occ | gt_occ).sum().float()

        iou = inter / (union + 1e-6)

        return iou

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        """
        Returns:
            inputs:          (B*T, C, H, W) for easy grid logging
            reconstructions: (B*T, C, H, W)
            samples:         (B*T, C, H, W)
        """
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)   # (B*T, C, H, W)
        b_t, c, h, w = x.shape
        xrec, posterior = self(x)                                   # (B*T,C,H,W)

        # batch 0
        gt_seq = x[:self.model.seq_len]         # (T,C,H,W)
        rec_seq = xrec[:self.model.seq_len]     # (T,C,H,W)

        # frame-wise IoU
        iou_list = []
        for ti in range(self.model.seq_len):
            iou_t = self.compute_iou(rec_seq[ti], gt_seq[ti], occ_thr=0.3)
            iou_list.append(iou_t.item())

        panel = torch.cat([gt_seq, rec_seq], dim=0)       # (2T,C,H,W)
        grid = make_grid(panel, nrow=10, normalize=False, value_range=(0, 1))

        grid_np = grid.detach().cpu().permute(1, 2, 0).numpy()

        if grid_np.shape[-1] == 1:
            grid_np = grid_np[..., 0]

        fig, ax = plt.subplots(figsize=(24, 8))
        ax.imshow(grid_np, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

        iou_text = "  ".join([f"t{ti+1}:{iou_list[ti]:.3f}" for ti in range(self.model.seq_len)])
        ax.set_title(f"Frame-wise IoU | {iou_text}", fontsize=12)

        log["GT | RECON | IoU"] = fig

        return log

class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x
