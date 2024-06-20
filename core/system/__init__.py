from .base import *
from .ddpm import *
from .multitask_ae_ddpm import MultitaskAE_DDPM

systems = {
    'ddpm': DDPM,
    'multitask_ae_ddpm': MultitaskAE_DDPM
}