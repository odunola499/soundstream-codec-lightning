import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Dict, Optional
from pqmf import PQMF
import numpy as np
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from omegaconf import DictConfig
from torch_stft import TorchSTFT


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}

def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax=None, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

class FeatureMatchLoss(nn.Module):
    def __init__(self):
        super(FeatureMatchLoss, self).__init__()

    def forward(self, real_features, fake_features):
        loss = 0
        num_items = 0
        for (fake_feature, real_feature) in zip(fake_features, real_features):
            if isinstance(fake_feature, list):
                for (_fake_feature, _real_feature) in zip(fake_feature, real_feature):
                    loss = loss + F.l1_loss(_fake_feature.float(), _real_feature.float().detach())
                    num_items += 1
            else:
                loss = loss + F.l1_loss(fake_feature.float(), real_feature.float().detach())
                num_items += 1
        loss /= num_items
        return loss


class LeastDLoss(nn.Module):
    def __init__(self):
        super(LeastDLoss, self).__init__()

    def forward(self, disc_outputs):
        loss = 0
        for dg in disc_outputs:
            dg = dg.float()
            l = torch.mean((1-dg)**2)
            loss += l
        return loss


class MSEDLoss(nn.Module):
    def __init__(self):
        super(MSEDLoss, self).__init__()
        self.loss_func = nn.MSELoss()

    def forward(self, score_fake, score_real):
        loss_real = self.loss_func(score_real, score_real.new_ones(score_real.shape))
        loss_fake = self.loss_func(score_fake, score_fake.new_zeros(score_fake.shape))
        loss_d = loss_real + loss_fake
        return loss_d, loss_real, loss_fake


class HingeDLoss(nn.Module):
    def __init__(self):
        super(HingeDLoss, self).__init__()

    def forward(self, score_fake, score_real):
        loss_real = torch.mean(F.relu(1. - score_real))
        loss_fake = torch.mean(F.relu(1. + score_fake))
        loss_d = loss_real + loss_fake
        return loss_d, loss_real, loss_fake


class MSEGLoss(nn.Module):
    def __init__(self):
        super(MSEGLoss, self).__init__()

    def forward(self, scores):
        loss_fake = 0
        num_items = 0
        if isinstance(scores, list):
            for score in scores:
                loss_fake = loss_fake + F.mse_loss(score, score.new_ones(score.shape))
                num_items += 1
        else:
            loss_fake = F.mse_loss(scores, scores.new_ones(scores.shape))
            num_items += 1
        return loss_fake / num_items


class HingeGLoss(nn.Module):
    def __init__(self):
        super(HingeGLoss, self).__init__()

    def forward(self, score_real):
        loss_fake = torch.mean(F.relu(1. - score_real))
        return loss_fake


def stft(x, fft_size, hop_size, win_size, window):
    x_stft = torch.stft(x, fft_size, hop_size, win_size, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    outputs = torch.clamp(real ** 2 + imag ** 2, min=1e-7).transpose(2, 1)
    outputs = torch.sqrt(outputs)

    return outputs


class SpectralConvergence(nn.Module):
    def __init__(self):
        super(SpectralConvergence, self).__init__()

    def forward(self, predicts_mag, targets_mag):
        x = torch.norm(targets_mag - predicts_mag, p='fro')
        y = torch.norm(targets_mag, p='fro')

        return x / y


class LogSTFTMagnitude(nn.Module):
    def __init__(self):
        super(LogSTFTMagnitude, self).__init__()

    def forward(self, predicts_mag, targets_mag):
        log_predicts_mag = torch.log(predicts_mag)
        log_targets_mag = torch.log(targets_mag)
        outputs = F.l1_loss(log_predicts_mag, log_targets_mag)

        return outputs


class STFTLoss(nn.Module):
    def __init__(
        self,
        fft_size=1024,
        hop_size=120,
        win_size=600,
    ):
        super(STFTLoss, self).__init__()

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer('window', torch.hann_window(win_size))
        self.sc_loss = SpectralConvergence()
        self.mag = LogSTFTMagnitude()

    def forward(self, predicts, targets):
        predicts_mag = stft(predicts, self.fft_size, self.hop_size, self.win_size, self.window)
        targets_mag = stft(targets, self.fft_size, self.hop_size, self.win_size, self.window)

        sc_loss = self.sc_loss(predicts_mag, targets_mag)
        mag_loss = self.mag(predicts_mag, targets_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        win_sizes=[600, 1200, 240],
        hop_sizes=[120, 240, 50],
        **kwargs
    ):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.loss_layers = torch.nn.ModuleList()
        for (fft_size, win_size, hop_size) in zip(fft_sizes, win_sizes, hop_sizes):
            self.loss_layers.append(STFTLoss(fft_size, hop_size, win_size))

    def forward(self, fake_signals, true_signals):
        sc_losses, mag_losses = [], []
        for layer in self.loss_layers:
            sc_loss, mag_loss = layer(fake_signals, true_signals)
            sc_losses.append(sc_loss)
            mag_losses.append(mag_loss)

        sc_loss = sum(sc_losses) / len(sc_losses)
        mag_loss = sum(mag_losses) / len(mag_losses)

        return sc_loss, mag_loss

class BasicDiscriminatorLoss(nn.Module):
    """Least-square GAN loss."""

    def __init__(self, config=None):
        super(BasicDiscriminatorLoss, self).__init__()

    def forward(self, real_outputs, fake_outputs):
        loss = 0
        real_losses = []
        fake_losses = []
        for dr, dg in zip(real_outputs, fake_outputs):
            dr = dr.float()
            dg = dg.float()
            real_loss = torch.mean((1-dr)**2)
            fake_loss = torch.mean(dg**2)
            loss += (real_loss + fake_loss)
            real_losses.append(real_loss.item())
            fake_losses.append(fake_loss.item())

        return loss


class MSEDiscriminatorLoss(BasicDiscriminatorLoss):
    def __init__(self, config=None):
        super().__init__(config)
        self.mse_loss = MSEDLoss()

    def apply_d_loss(self, scores_fake, scores_real, loss_func):
        total_loss = 0
        total_real_loss = 0
        total_fake_loss = 0
        if isinstance(scores_fake, list):
            # multi-scale loss
            for score_fake, score_real in zip(scores_fake, scores_real):
                loss, real_loss, fake_loss = loss_func(score_fake=score_fake, score_real=score_real)
                total_loss = total_loss + loss
                total_real_loss = total_real_loss + real_loss
                total_fake_loss = total_fake_loss + fake_loss
            # normalize loss values with number of scales
            total_loss /= len(scores_fake)
            total_real_loss /= len(scores_real)
            total_fake_loss /= len(scores_fake)
        else:
            # single scale loss
            total_loss, total_real_loss, total_fake_loss = loss_func(scores_fake, scores_real)
        return total_loss, total_real_loss, total_fake_loss

    def forward(self, real_scores, fake_scores):
        mse_D_loss, mse_D_real_loss, mse_D_fake_loss = self.apply_d_loss(
            scores_fake=fake_scores,
            scores_real=real_scores,
            loss_func=self.mse_loss)
        return mse_D_loss





def freq_MAE(estimation, target, win=2048, stride=512, srs=None, sudo_sr=None):
    est_spec = torch.stft(
        estimation.view(-1, estimation.shape[-1]),
        n_fft=win,
        hop_length=stride,
        window=torch.hann_window(win).to(estimation.device).float(),
        return_complex=True,
    )
    est_target = torch.stft(
        target.view(-1, target.shape[-1]),
        n_fft=win,
        hop_length=stride,
        window=torch.hann_window(win).to(estimation.device).float(),
        return_complex=True,
    )

    if srs is None:
        return (est_spec.real - est_target.real).abs().mean() + (est_spec.imag - est_target.imag).abs().mean()
    else:
        loss = 0
        for i, sr in enumerate(srs):
            max_freq = int(est_spec.shape[-2] * sr / sudo_sr) + 1
            loss += (est_spec[i][:max_freq].real - est_target[i][:max_freq].real).abs().mean() \
                    + (est_spec[i][:max_freq].imag - est_target[i][:max_freq].imag).abs().mean()
        loss = loss / len(srs)
        # import pdb; pdb.set_trace()
        return loss


def wav_MAE(ests, refs):
    return torch.mean(torch.abs(ests - refs))


def sisnr(x, s, eps=1e-8):
    '''
    Calculate si-snr loss
    x: Bsz*T ests
    s: Bsz*T refs
    '''

    x, s = x.view(-1, x.shape[-1]), s.view(-1, s.shape[-1])

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimension mismatch when calculate si-snr, {} vs {}".format(x.shape, s.shape)
        )
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True) ** 2 + eps)

    return - 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)).mean()


def snr(x, s, eps=1e-8):
    '''
    Calculate si-snr loss
    x: Bsz*T ests
    s: Bsz*T refs
    '''

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimension mismatch when calculate si-snr, {} vs {}".format(x.shape, s.shape)
        )
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    return - 20 * torch.log10(l2norm(s_zm) / (l2norm(x_zm - s_zm) + eps) + eps).mean()


def mel_MAE(ests, refs, sr=48000, n_fft=2048, hop_length=512, n_mels=80):
    compute_Melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    ests_melspec = compute_Melspec(ests)
    refs_melspec = compute_Melspec(refs)

    return (ests_melspec - refs_melspec).abs().mean()


# adapted from ENHANCE-PASS
class BasicEnhancementLoss(nn.Module):
    """
    Config:
        sr: sample_rate
        loss_type: List[str]

    """

    def __init__(self, config):
        super(BasicEnhancementLoss, self).__init__()
        self.sr = config.sr
        self.loss_type = config.loss_type
        self.win = config.win
        self.stride = config.stride

        loss_weight = config.loss_weight
        if loss_weight == None:
            self.loss_weight = [1.0] * len(self.loss_type)
        else:
            self.loss_weight = loss_weight

        if 'mel_MAE' in self.loss_type:
            self.compute_Melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr, n_fft=self.win, hop_length=self.stride, n_mels=80)
            # self.compute_Melspec.to(device)

    def mel_MAE(self, ests, refs):
        ests_melspec = self.compute_Melspec(ests)
        refs_melspec = self.compute_Melspec(refs)

        return (ests_melspec - refs_melspec).abs().mean()

    def __call__(self, ests, refs, wav_lens=None, srs=None):
        loss_dic = {}
        loss = 0
        for i, item in enumerate(self.loss_type):
            # import pdb; pdb.set_trace()
            if item == 'freq_MAE':
                loss_dic[item] = eval(item)(
                    ests, refs, win=self.win, stride=self.stride, srs=srs, sudo_sr=self.sr)
            elif item == 'mel_MAE':
                loss_dic[item] = self.mel_MAE(ests, refs)
                # import pdb; pdb.set_trace()
            else:
                # wave MAE
                loss_dic[item] = eval(item)(ests, refs)
            if self.loss_weight[i] > 0:
                loss = self.loss_weight[i] * loss_dic[item] + loss

        return loss, loss_dic


class BasicGeneratorLoss(nn.Module):
    def __init__(self, config):
        super(BasicGeneratorLoss, self).__init__()
        self.config = config
        self.adv_criterion = eval(config.adv_criterion)()
        if self.config.use_feature_match:
            self.feature_match_criterion = FeatureMatchLoss()

    def forward(
            self,
            targets: torch.Tensor,
            outputs: torch.Tensor,
            output_real: Dict[str, torch.Tensor],
            output_fake: Dict[str, torch.Tensor],
            fmap_real: Optional[Dict[str, torch.Tensor]] = None,
            fmap_fake: Optional[Dict[str, torch.Tensor]] = None,
            use_adv_loss: bool = True,
    ):
        """
        Args:
            targets: ground-truth waveforms.
            outputs: generated waveforms.
            output_real: logits from discriminators on real waveforms.
            output_fake: logits from discriminators on generated/fake waveforms.
            fmap_real: feature mappings of real waveforms.
            fmap_fake: feature mappings of generated/fake waveforms.
        """
        g_loss = 0
        g_loss_items = {}

        if use_adv_loss:
            for key in output_fake.keys():
                adv_loss_item = self.adv_criterion(output_fake[key])
                g_loss += adv_loss_item
                g_loss_items[f"Train/G_adv_{key}"] = adv_loss_item.item()

                if self.config.use_feature_match:
                    assert fmap_real is not None and fmap_fake is not None
                    fmap_loss_item = self.feature_match_criterion(
                        fmap_real[key], fmap_fake[key]) * self.config.feat_match_loss_weight
                    g_loss += fmap_loss_item
                    g_loss_items[f"Train/G_fm_{key}"] = fmap_loss_item.item() / self.config.feat_match_loss_weight

        if self.config.use_mel_loss:
            hps_mel_scale_loss = self.config.mel_scale_loss if isinstance(self.config.mel_scale_loss, list) \
                else [self.config.mel_scale_loss]

            for i, _hps_mel_scale_loss in enumerate(hps_mel_scale_loss):
                outputs_mel = mel_spectrogram(outputs.squeeze(1), **_hps_mel_scale_loss)
                target_mel = mel_spectrogram(targets.squeeze(1), **_hps_mel_scale_loss)
                mel_loss = F.l1_loss(outputs_mel, target_mel.detach()) * self.config.mel_loss_weight
                g_loss += mel_loss
                g_loss_items[f"Train/G_mel_loss_{i}"] = mel_loss.item() / self.config.mel_loss_weight

        return g_loss, g_loss_items


class GeneratorSTFTLoss(BasicGeneratorLoss):
    def __init__(self, config):
        super().__init__(config)
        if self.config.use_full_stft_loss:
            self.stft_full_criterion = MultiResolutionSTFTLoss(
                **self.config.full_multi_scale_stft_loss)

        if self.config.use_sub_stft_loss:
            self.pqmf = PQMF(self.config.sub_multi_scale_stft_loss.num_bands)
            self.stft_sub_criterion = MultiResolutionSTFTLoss(
                **self.config.sub_multi_scale_stft_loss)

    def forward(
            self, targets, outputs, output_real, output_fake, fmap_real, fmap_fake,
            use_adv_loss: bool = True
    ):
        g_loss, g_loss_items = super().forward(
            targets, outputs, output_real, output_fake, fmap_real, fmap_fake, use_adv_loss=use_adv_loss)

        # Optional: full-band STFT Loss
        if self.config.use_full_stft_loss:
            sc_full_loss, mg_full_loss = \
                self.stft_full_criterion(outputs.squeeze(1), targets.squeeze(1))
            g_loss = g_loss + self.config.full_stft_loss_weight * (sc_full_loss + mg_full_loss)
            g_loss_items["Train/G_sc_full"] = sc_full_loss.item()
            g_loss_items["Train/G_mg_full"] = mg_full_loss.item()

        # Optional: sub-band STFT Loss
        if self.config.use_sub_stft_loss:
            targets_sub = self.pqmf.analysis(targets)
            outputs_sub = self.pqmf.analysis(outputs)
            size = outputs_sub.size(-1)
            outputs_sub_view = outputs_sub.view(-1, size)
            targets_sub_view = targets_sub.view(-1, size)

            sc_sub_loss, mg_sub_loss = \
                self.stft_sub_criterion(outputs_sub_view, targets_sub_view)
            g_loss = g_loss + self.config.sub_stft_loss_weight * (sc_sub_loss + mg_sub_loss)
            g_loss_items["Train/G_sc_sub"] = sc_sub_loss.item()
            g_loss_items["Train/G_mg_sub"] = mg_sub_loss.item()

        return g_loss, g_loss_items


class GeneratorSTFTEnhLoss(GeneratorSTFTLoss):
    def __init__(self, config):
        super().__init__(config)
        self.enh_criterion = BasicEnhancementLoss(config.enh_loss)

    def forward(self, targets, outputs, output_real, output_fake, fmap_real, fmap_fake, use_adv_loss: bool = True):
        g_loss, g_loss_items = super().forward(
            targets, outputs, output_real, output_fake, fmap_real, fmap_fake, use_adv_loss=use_adv_loss)

        enh_loss, enh_loss_items = self.enh_criterion(outputs, targets)
        g_loss = g_loss + enh_loss
        for k, v in enh_loss_items.items():
            g_loss_items[f"Train/{k}"] = v.item()

        return g_loss, g_loss_items

class BasicDiscriminatorLoss(nn.Module):
    """Least-square GAN loss."""

    def __init__(self, config=None):
        super(BasicDiscriminatorLoss, self).__init__()

    def forward(self, real_outputs, fake_outputs):
        loss = 0
        real_losses = []
        fake_losses = []
        for dr, dg in zip(real_outputs, fake_outputs):
            dr = dr.float()
            dg = dg.float()
            real_loss = torch.mean((1-dr)**2)
            fake_loss = torch.mean(dg**2)
            loss += (real_loss + fake_loss)
            real_losses.append(real_loss.item())
            fake_losses.append(fake_loss.item())

        return loss

class MSEDiscriminatorLoss(BasicDiscriminatorLoss):
    def __init__(self, config=None):
        super().__init__(config)
        self.mse_loss = MSEDLoss()

    def apply_d_loss(self, scores_fake, scores_real, loss_func):
        total_loss = 0
        total_real_loss = 0
        total_fake_loss = 0
        if isinstance(scores_fake, list):
            # multi-scale loss
            for score_fake, score_real in zip(scores_fake, scores_real):
                loss, real_loss, fake_loss = loss_func(score_fake=score_fake, score_real=score_real)
                total_loss = total_loss + loss
                total_real_loss = total_real_loss + real_loss
                total_fake_loss = total_fake_loss + fake_loss
            # normalize loss values with number of scales
            total_loss /= len(scores_fake)
            total_real_loss /= len(scores_real)
            total_fake_loss /= len(scores_fake)
        else:
            # single scale loss
            total_loss, total_real_loss, total_fake_loss = loss_func(scores_fake, scores_real)
        return total_loss, total_real_loss, total_fake_loss

    def forward(self, real_scores, fake_scores):
        mse_D_loss, mse_D_real_loss, mse_D_fake_loss = self.apply_d_loss(
            scores_fake=fake_scores,
            scores_real=real_scores,
            loss_func=self.mse_loss)
        return mse_D_loss


class MultiFrequencyDiscriminator(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.stfts = nn.ModuleList([
            TorchSTFT(
                fft_size=x * 4,
                hop_size=x,
                win_size=x * 4,
                normalized=True,  # returns the normalized STFT results, i.e., multiplied by frame_length^{-0.5}
                domain=config.domain,
                mel_scale=config.mel_scale,
                sample_rate=config.sample_rate,
            ) for x in config.hop_lengths
        ])

        self.domain = config.domain
        if self.domain == 'double':
            self.discriminators = nn.ModuleList([
                FrequenceDiscriminator(2, c)
                for x, c in zip(config.hop_lengths, config.hidden_channels)])
        else:
            self.discriminators = nn.ModuleList([
                FrequenceDiscriminator(1, c)
                for x, c in zip(config.hop_lengths, config.hidden_channels)])

    def forward(self, y, y_hat, **kwargs):
        if y.ndim == 3:
            y = y.view(-1, y.shape[-1])

        if y_hat.ndim == 3:
            y_hat = y_hat.view(-1, y_hat.shape[-1])

        real_outputs = []
        fake_outputs = []
        real_feature_maps = []
        fake_feature_maps = []

        for stft, layer in zip(self.stfts, self.discriminators):
            mag, phase = stft.transform(y.squeeze(1))
            fake_mag, fake_phase = stft.transform(y_hat.squeeze(1))
            if self.domain == 'double':
                mag = torch.stack(torch.chunk(mag, 2, dim=1), dim=1)
                fake_mag = torch.stack(torch.chunk(fake_mag, 2, dim=1), dim=1)
            else:
                mag = mag.unsqueeze(1)
                fake_mag = fake_mag.unsqueeze(1)

            real_out, real_feat_map = layer(mag)
            fake_out, fake_feat_map = layer(fake_mag)
            real_outputs.append(real_out)
            fake_outputs.append(fake_out)
            real_feature_maps.append(real_feat_map)
            fake_feature_maps.append(fake_feat_map)

        return real_outputs, fake_outputs, real_feature_maps, fake_feature_maps


class FrequenceDiscriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels=512):
        super(FrequenceDiscriminator, self).__init__()

        self.discriminator = nn.ModuleList()
        self.discriminator += [
            nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    in_channels, hidden_channels // 32,
                    kernel_size=(3, 3), stride=(1, 1)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 32, hidden_channels // 16,
                    kernel_size=(3, 3), stride=(2, 2)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 16, hidden_channels // 8,
                    kernel_size=(3, 3), stride=(1, 1)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 8, hidden_channels // 4,
                    kernel_size=(3, 3), stride=(2, 2)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 4, hidden_channels // 2,
                    kernel_size=(3, 3), stride=(1, 1)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 2, hidden_channels,
                    kernel_size=(3, 3), stride=(2, 2)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels, 1,
                    kernel_size=(3, 3), stride=(1, 1)))
            )
        ]

    def forward(self, x):
        hiddens = []
        for layer in self.discriminator:
            x = layer(x)
            hiddens.append(x)
        return x, hiddens[:-1]