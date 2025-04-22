# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from unittest.mock import MagicMock, patch

from nemo_reinforcer.algorithms.dpo import add_ref_logprobs_to_data, dpo_train


class MockPolicy:
    def __init__(self, logprobs):
        self.logprobs = logprobs

    def get_reference_policy_logprobs(self, batch, micro_batch_size):
        return {"reference_logprobs": self.logprobs}

    def train(self, batch, loss_fn, eval_mode=False, gbs=None, mbs=None):
        # Return mock training results
        return {"loss": torch.tensor(0.0), "all_mb_metrics": {}}

    def prepare_for_training(self):
        pass


def test_add_logprobs_to_batch():
    """Test that add_ref_logprobs_to_data correctly adds reference policy logprobs to batches."""
    # Create mock data
    batch_size = 2
    seq_len = 4
    vocab_size = 16

    # Create a mock batch
    mock_batch = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
    }

    # Create mock logprobs that will be returned by the policy
    mock_logprobs = torch.randn(batch_size, seq_len)

    # Create a mock dataloader that yields our mock batch
    mock_dataloader = MagicMock()
    mock_dataloader.__iter__.return_value = iter([mock_batch])

    # Create a mock policy that returns our mock logprobs
    mock_policy = MockPolicy(mock_logprobs)

    # Create a mock master config
    mock_master_config = {"policy": {"train_micro_batch_size": 1}}

    # Get the augmented batches
    augmented_batches = list(
        add_ref_logprobs_to_data(mock_dataloader, mock_policy, mock_master_config)
    )

    # Verify we got exactly one batch
    assert len(augmented_batches) == 1
    augmented_batch = augmented_batches[0]

    # Verify the original batch data is preserved
    assert torch.equal(augmented_batch["input_ids"], mock_batch["input_ids"])
    assert torch.equal(augmented_batch["attention_mask"], mock_batch["attention_mask"])

    # Verify the reference policy logprobs were added correctly
    assert "reference_policy_logprobs" in augmented_batch
    assert augmented_batch["reference_policy_logprobs"].shape == (batch_size, seq_len)

    # Verify the logprobs were rolled by -1 as expected
    expected_logprobs = torch.roll(mock_logprobs, -1, dims=-1)
    assert torch.equal(augmented_batch["reference_policy_logprobs"], expected_logprobs)


def test_dpo_data_iteration_order_changes_between_epochs():
    """Test that DPO iterates over data in a different order each epoch."""
    # Create mock data
    batch_size = 2
    seq_len = 4
    vocab_size = 16
    num_batches = 3  # Small number of batches for testing
    max_num_epochs = 2
    # Create mock batches
    mock_examples = []
    for i in range(num_batches * batch_size):
        mock_example = {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
        }
        mock_examples.append(mock_example)

    # Create a custom dataset class to yield our mock batches
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, batches):
            self.batches = batches

        def __len__(self):
            return len(self.batches)

        def __getitem__(self, idx):
            return self.batches[idx]

    # Create the dataset
    dataset = TestDataset(mock_examples)

    # Create mock logprobs that will be returned by the policy
    dummy_logprobs = torch.randn(batch_size, seq_len)

    # Create a mock policy that returns our mock logprobs
    mock_policy = MockPolicy(dummy_logprobs)

    # Create a mock master config
    mock_master_config = {
        "policy": {
            "train_micro_batch_size": 1,
            "train_global_batch_size": batch_size,
            "make_sequence_length_divisible_by": 1,
        },
        "dpo": {
            "seed": 42,  # Fixed seed for reproducibility
            "max_num_epochs": max_num_epochs,
            "max_num_steps": 10000,  # set to a large number to allow for 2 epochs
            "val_period": 0,  # Disable validation for this test
            "val_batches": 0,
            "val_global_batch_size": 0,
            "val_micro_batch_size": 0,
            "val_at_start": False,
        },
        "checkpointing": {"enabled": False},
    }

    # Create mock logger
    mock_logger = MagicMock()
    mock_logger.log_metrics = MagicMock()

    # Create mock checkpointer
    mock_checkpointer = MagicMock()
    mock_checkpointer.get_latest_checkpoint_path.return_value = None

    # Track the order of batches seen in each epoch
    data_orders = []

    # Create a wrapper around the policy's train method to track batch order
    original_train = mock_policy.train

    def track_batch_order_train(*args, **kwargs):
        batch = args[0]
        data_orders.append(batch["input_ids"].sum().item())
        return original_train(*args, **kwargs)

    mock_policy.train = track_batch_order_train

    def create_dataloader():
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )
        return dataloader

    # Run dpo_train
    with patch("nemo_reinforcer.algorithms.dpo.validate", return_value=({}, {})):
        train_dataloader = create_dataloader()

        # Run dpo_train
        dpo_train(
            policy=mock_policy,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            tokenizer=None,
            loss_fn=MagicMock(),
            master_config=mock_master_config,
            logger=mock_logger,
            checkpointer=mock_checkpointer,
            dpo_save_state=None,
        )

    assert len(data_orders) == max_num_epochs * num_batches
    # Verify that the order of batches is different between epochs
    assert data_orders[0:num_batches] != data_orders[num_batches : 2 * num_batches], (
        "Data iteration order should be different between epochs"
    )
    # Verify that the cumulative data seen is the same in both epochs
    assert sum(data_orders[0:num_batches]) == sum(
        data_orders[num_batches : 2 * num_batches]
    ), "All batches should be seen in both epochs"
