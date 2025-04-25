# Loss functions in NeMo-RL

Loss functions in NeMo-RL are specially designed to ensure that full batch training is equivalent to training with gradient accumulation. To understand
why special care needs to be taken here, consider the following example of a simple loss function that takes the average of some loss over all tokens in a microbatch, and then averages loss over the microbatches.

Suppose we have a global batch with 16 unmasked tokens. The first 10 unmasked tokens come from the first half of the samples in the batch, and the last 6 come from the second half. If training with one global batch,

$$
L = \frac{\sum_{t=1}^{16} L_t}{16}.
$$

But if we train with two microbatches, 

$$
L = \frac{\frac{\sum_{t=1}^{10} L_t}{10} + \frac{\sum_{t={10}}^{16} L_t}{6}}{2},
$$

which is, in general, not equivalent to the full-batch loss. To fix this, we need each microbatch to have information about how many tokens are in the other microbatches in the global batch.

In NeMo-RL, this information is passed to the loss function directly. Each loss function is expected to fall into one of two categories, token-level or sample-level, which is an attribute of the loss function itself (see [loss_functions.py](../../nemo_reinforcer/algorithms/loss_functions.py) for some examples). The policy then uses this information to compute the global normalization factor using the full batch (for token-level losses, this is the total number of tokens in the batch. For sample-level losses, this is the number of valid samples in the batch). The normalization factor is then passed to the loss function, which uses it to normalize the microbatch loss. To get the loss for the global batch, the policy simply sums across all microbatch losses.

For our simple example above, this would look like:

```python
from typing import Tuple

import torch
from nemo_reinforcer.algorithms.interfaces import LossFunction
from nemo_reinforcer.algorithms.loss_functions import LossType
from nemo_reinforcer.distributed.batched_data_dict import BatchedDataDict


class SimpleAverageLoss(LossFunction):
    """Simple average loss function that demonstrates proper microbatch handling.
    
    NOTE: We assume for simplicity that the losses per token are passed directly into the this loss function.
          This is not the case in practice!
    """

    loss_type = LossType.TOKEN_LEVEL

    def __call__(
        self,
        next_token_losses: torch.Tensor,
        data: BatchedDataDict[SimpleAverageLossDataDict],
        normalization_factor: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the simple average loss with proper microbatch handling."""
        token_mask = data["token_mask"] ## token mask for this microbatch
        sample_mask = data["sample_mask"] ## sample mask for this microbatch

        # mask.sum() will be 10 for microbatch 1, 6 for microbatch 2
        mask = token_mask * sample_mask.unsqueeze(-1)

        # normalization_factor will be 16 in our example since there are 16 tokens in the global batch
        # since we specified that this is a token-level loss, the policy
        # will give us the right normalization factor automatically.
        loss = (next_token_losses * mask).sum() / (normalization_factor + 1e-8)
        return loss
```