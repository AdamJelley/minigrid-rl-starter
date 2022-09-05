import random
import numpy
import torch
import collections
import os
import GPUtil


# Set device
if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    deviceID = GPUtil.getFirstAvailable(
        order="load",
        maxLoad=0.4,
        maxMemory=0.4,
        attempts=1,
        interval=900,
        verbose=False,
    )[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceID)
    device = torch.device(f"cuda:{deviceID}")
# elif torch.has_mps:
#     device = torch.device("mps")
else:
    device = torch.device("cpu")


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d
