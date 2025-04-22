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
from unittest.mock import MagicMock
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from nemo_reinforcer.algorithms.sft import sft_train, _default_sft_save_state
from nemo_reinforcer.algorithms.loss_functions import NLLLoss


@pytest.fixture
def mock_components():
    # Create mock components
    policy = MagicMock()
    policy.train.return_value = {"loss": torch.tensor(0.5), "all_mb_metrics": {}}

    # Create a proper message log structure with token_ids
    mock_batch = {
        "message_log": [[{"token_ids": torch.tensor([1, 2, 3]), "role": "assistant"}]],
        "loss_multiplier": torch.tensor(1.0),
    }

    # Create mock dataloader with 10 batches that can be iterated multiple times
    train_dataloader = MagicMock(spec=StatefulDataLoader)

    def train_iter(self):
        return iter([mock_batch] * 10)

    train_dataloader.__iter__ = train_iter
    train_dataloader.__len__ = MagicMock(return_value=10)

    val_dataloader = MagicMock(spec=StatefulDataLoader)

    def val_iter(self):
        return iter([mock_batch] * 10)

    val_dataloader.__iter__ = val_iter
    val_dataloader.__len__ = MagicMock(return_value=10)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    loss_fn = NLLLoss()
    logger = MagicMock()
    checkpointer = MagicMock()
    sft_task_spec = MagicMock()

    # Create mock master config
    master_config = {
        "sft": {
            "max_num_steps": 5,
            "max_num_epochs": 2,
            "val_period": 100,
            "val_batches": 1,
            "val_global_batch_size": 1,
            "val_micro_batch_size": 1,
            "val_at_start": False,
        },
        "policy": {
            "train_global_batch_size": 1,
            "make_sequence_length_divisible_by": 8,
        },
        "checkpointing": {"enabled": False},
    }

    return {
        "policy": policy,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "tokenizer": tokenizer,
        "loss_fn": loss_fn,
        "logger": logger,
        "checkpointer": checkpointer,
        "sft_task_spec": sft_task_spec,
        "master_config": master_config,
    }


def test_exit_on_max_steps(mock_components):
    """Test that training loop exits when max_num_steps is reached"""
    # Set max steps to 12, which is less than len(train_dataloader) * max_num_epochs
    mock_components["master_config"]["sft"]["max_num_steps"] = 12

    sft_save_state = _default_sft_save_state()

    # Run training
    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["sft_task_spec"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    # Verify we only trained for 12 steps
    assert mock_components["policy"].train.call_count == 12


def test_exit_on_max_epochs(mock_components):
    """Test that training loop exits when max_num_epochs is reached"""
    # Set max epochs to 2 and max steps to a large number
    mock_components["master_config"]["sft"]["max_num_epochs"] = 2
    mock_components["master_config"]["sft"]["max_num_steps"] = 100

    sft_save_state = _default_sft_save_state()

    # Run training
    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["sft_task_spec"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    # Verify we trained for exactly two epochs (20 batches)
    assert mock_components["policy"].train.call_count == 20


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
from unittest.mock import MagicMock
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from nemo_reinforcer.algorithms.sft import sft_train, _default_sft_save_state
from nemo_reinforcer.algorithms.loss_functions import NLLLoss
from unittest.mock import patch


class MockPolicy:
    def __init__(self, logprobs):
        self.logprobs = logprobs

    def train(self, batch, loss_fn, eval_mode=False, gbs=None, mbs=None):
        # Return mock training results
        return {"loss": torch.tensor(0.0), "all_mb_metrics": {}}

    def prepare_for_training(self):
        pass


@pytest.fixture
def mock_components():
    # Create mock components
    policy = MagicMock()
    policy.train.return_value = {"loss": torch.tensor(0.5), "all_mb_metrics": {}}

    # Create a proper message log structure with token_ids
    mock_batch = {
        "message_log": [[{"token_ids": torch.tensor([1, 2, 3]), "role": "assistant"}]],
        "loss_multiplier": torch.tensor(1.0),
    }

    # Create mock dataloader with 10 batches that can be iterated multiple times
    train_dataloader = MagicMock(spec=StatefulDataLoader)

    def train_iter(self):
        return iter([mock_batch] * 10)

    train_dataloader.__iter__ = train_iter
    train_dataloader.__len__ = MagicMock(return_value=10)

    val_dataloader = MagicMock(spec=StatefulDataLoader)

    def val_iter(self):
        return iter([mock_batch] * 10)

    val_dataloader.__iter__ = val_iter
    val_dataloader.__len__ = MagicMock(return_value=10)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    loss_fn = NLLLoss()
    logger = MagicMock()
    checkpointer = MagicMock()
    sft_task_spec = MagicMock()

    # Create mock master config
    master_config = {
        "sft": {
            "max_num_steps": 5,
            "max_num_epochs": 2,
            "val_period": 100,
            "val_batches": 1,
            "val_global_batch_size": 1,
            "val_micro_batch_size": 1,
            "val_at_start": False,
        },
        "policy": {
            "train_global_batch_size": 1,
            "make_sequence_length_divisible_by": 8,
        },
        "checkpointing": {"enabled": False},
    }

    return {
        "policy": policy,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "tokenizer": tokenizer,
        "loss_fn": loss_fn,
        "logger": logger,
        "checkpointer": checkpointer,
        "sft_task_spec": sft_task_spec,
        "master_config": master_config,
    }


def test_exit_on_max_steps(mock_components):
    """Test that training loop exits when max_num_steps is reached"""
    # Set max steps to 12, which is less than len(train_dataloader) * max_num_epochs
    mock_components["master_config"]["sft"]["max_num_steps"] = 12

    sft_save_state = _default_sft_save_state()

    # Run training
    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["sft_task_spec"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    # Verify we only trained for 12 steps
    assert mock_components["policy"].train.call_count == 12


def test_exit_on_max_epochs(mock_components):
    """Test that training loop exits when max_num_epochs is reached"""
    # Set max epochs to 2 and max steps to a large number
    mock_components["master_config"]["sft"]["max_num_epochs"] = 2
    mock_components["master_config"]["sft"]["max_num_steps"] = 100

    sft_save_state = _default_sft_save_state()

    # Run training
    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["sft_task_spec"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    # Verify we trained for exactly two epochs (20 batches)
    assert mock_components["policy"].train.call_count == 20


def test_sft_data_iteration_order_changes_between_epochs():
    """Test that SFT iterates over data in a different order each epoch."""
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
            "message_log": [
                [
                    {"token_ids": torch.tensor([i] * seq_len), "role": "user"},
                    {"token_ids": torch.tensor([i] * seq_len), "role": "assistant"},
                ]
            ],
            "loss_multiplier": torch.tensor(1.0),
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

    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    # Create a mock master config
    mock_master_config = {
        "policy": {
            "train_micro_batch_size": 1,
            "train_global_batch_size": batch_size,
            "make_sequence_length_divisible_by": 1,
        },
        "sft": {
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
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    # Run sft_train
    with patch("nemo_reinforcer.algorithms.sft.validate", return_value=({}, {})):
        train_dataloader = create_dataloader()

        # Run sft_train
        sft_train(
            policy=mock_policy,
            train_dataloader=train_dataloader,
            val_dataloader=None,
            tokenizer=mock_tokenizer,
            loss_fn=MagicMock(),
            master_config=mock_master_config,
            logger=mock_logger,
            sft_task_spec=MagicMock(),
            checkpointer=mock_checkpointer,
            sft_save_state=None,
        )

    # Verify that we have seen all batches in both epochs
    assert len(data_orders) == max_num_epochs * num_batches, (
        "Should have seen all batches in both epochs"
    )

    print(data_orders)
    # Verify that the order of batches is different between epochs
    assert data_orders[0:num_batches] != data_orders[num_batches : 2 * num_batches], (
        "Data iteration order should be different between epochs"
    )

    # Verify that the cumulative data seen is the same in both epochs
    assert sum(data_orders[0:num_batches]) == sum(
        data_orders[num_batches : 2 * num_batches]
    ), "All batches should be seen in both epochs"
