import jax
from flax.struct import dataclass, field
from flax import linen as nn
from jaxed.GP.kernel import Kernel

# make it get a kernel
# it should also have weights that are initialized somehow
# then they can be 'fit' to train data, in some training loop
class GaussianProcess(nn.Module):
    pass
