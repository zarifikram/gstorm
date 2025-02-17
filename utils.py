import torch
import os
import numpy as np
import random
from tensorboardX import SummaryWriter
from einops import repeat
from contextlib import contextmanager
import time
import yacs
from yacs.config import CfgNode as CN

def create_cartesian_product_examples(sequence_length):
    # precompute timestep pairs (t, t+k) for the forward and backward encoders
    ft = torch.arange(sequence_length, dtype=torch.int32) # [0, 1 .... T-1]
    dt = torch.arange(2,sequence_length, dtype=torch.int32)
    combinations = torch.cartesian_prod(ft, dt)

    combinations[:, 1] = combinations[:, 0] + combinations[:, 1]
    fb_pairs = combinations.clone()

    #make sure no fb_pair goes over limit.
    fb_pairs = fb_pairs[combinations[:,1] < sequence_length]
    return fb_pairs, fb_pairs[:,1] - fb_pairs[:, 0]
    
def seed_np_torch(seed=20001118):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger():
    def __init__(self, path) -> None:
        self.writer = SummaryWriter(logdir=path, flush_secs=1)
        self.tag_step = {}

    def log(self, tag, value):
        if tag not in self.tag_step:
            self.tag_step[tag] = 0
        else:
            self.tag_step[tag] += 1
        if "video" in tag:
            self.writer.add_video(tag, value, self.tag_step[tag], fps=15)
        elif "images" in tag:
            self.writer.add_images(tag, value, self.tag_step[tag])
        elif "hist" in tag:
            self.writer.add_histogram(tag, value, self.tag_step[tag])
        else:
            self.writer.add_scalar(tag, value, self.tag_step[tag])


class EMAScalar():
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


def load_config(config_path):
    conf = CN()
    # Task need to be RandomSample/TrainVQVAE/TrainWorldModel
    conf.Task = ""

    conf.BasicSettings = CN()
    conf.BasicSettings.Seed = 0
    conf.BasicSettings.ImageSize = 0
    conf.BasicSettings.ReplayBufferOnGPU = False

    # Under this setting, input 128*128 -> latent 16*16*64
    conf.Models = CN()

    conf.Models.WorldModel = CN()
    conf.Models.WorldModel.InChannels = 0
    conf.Models.WorldModel.TransformerMaxLength = 0
    conf.Models.WorldModel.TransformerHiddenDim = 0
    conf.Models.WorldModel.TransformerNumLayers = 0
    conf.Models.WorldModel.TransformerNumHeads = 0

    conf.Models.Agent = CN()
    conf.Models.Agent.NumLayers = 0
    conf.Models.Agent.HiddenDim = 256
    conf.Models.Agent.Gamma = 1.0
    conf.Models.Agent.Lambda = 0.0
    conf.Models.Agent.EntropyCoef = 0.0

    conf.JointTrainAgent = CN()
    conf.JointTrainAgent.SampleMaxSteps = 0
    conf.JointTrainAgent.BufferMaxLength = 0
    conf.JointTrainAgent.BufferWarmUp = 0
    conf.JointTrainAgent.NumEnvs = 0
    conf.JointTrainAgent.BatchSize = 0
    conf.JointTrainAgent.DemonstrationBatchSize = 0
    conf.JointTrainAgent.BatchLength = 0
    conf.JointTrainAgent.ImagineBatchSize = 0
    conf.JointTrainAgent.ImagineDemonstrationBatchSize = 0
    conf.JointTrainAgent.ImagineContextLength = 0
    conf.JointTrainAgent.ImagineBatchLength = 0
    conf.JointTrainAgent.TrainDynamicsEverySteps = 0
    conf.JointTrainAgent.TrainAgentEverySteps = 0
    conf.JointTrainAgent.SaveEverySteps = 0
    conf.JointTrainAgent.UseDemonstration = False

    conf.Models.MaskGit = CN()
    conf.Models.MaskGit.VocabSize = 0
    conf.Models.MaskGit.VocabDim = 128
    conf.Models.MaskGit.T_revise = 0
    conf.Models.MaskGit.T_draft = 0
    conf.Models.MaskGit.M = 0
    conf.Models.MaskGit.MaskSchedule = ""
    conf.Models.MaskGit.TmfArgs = CN()
    conf.Models.MaskGit.TmfArgs.EmbedDim = 128
    conf.Models.MaskGit.TmfArgs.NumHeads = 0
    conf.Models.MaskGit.TmfArgs.NumLayers = 0
    conf.Models.MaskGit.TmfArgs.Dropout = 0.0
    conf.Models.MaskGit.TmfArgs.AttentionDropout = 0.0
    conf.Models.MaskGit.TmfArgs.MlpDim = 0


    conf.defrost()
    conf.set_new_allowed(True)
    conf.merge_from_file(config_path)
    conf.freeze()

    return conf
