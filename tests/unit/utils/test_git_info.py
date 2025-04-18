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

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from nemo_reinforcer.utils.git_info import get_git_info, get_git_diff


def test_get_git_info_success():
    """Test get_git_info when git commands succeed."""
    mock_commit = "abc123"
    mock_branch = "test_branch"

    with patch("subprocess.run") as mock_run:
        # Mock the commit command
        mock_commit_output = MagicMock()
        mock_commit_output.stdout = f"{mock_commit}\n"

        # Mock the branch command
        mock_branch_output = MagicMock()
        mock_branch_output.stdout = f"{mock_branch}\n"

        # Return the mock commit the first time subprocess.run is called
        # and the mock branch the second time
        mock_run.side_effect = [mock_commit_output, mock_branch_output]

        commit, branch = get_git_info()

        assert commit == mock_commit
        assert branch == mock_branch


def test_get_git_info_error():
    """Test get_git_info when git commands fail."""
    with patch("subprocess.run", side_effect=OSError("git not found")):
        commit, branch = get_git_info()
        assert commit == ""
        assert branch == ""


def test_get_git_diff_success():
    """Test get_git_diff when git command succeeds."""
    mock_diff = "diff --git a/file.txt b/file.txt\n+++ b/file.txt\n@@ -1,1 +1,2 @@"

    with patch("subprocess.run") as mock_run:
        mock_output = MagicMock()
        mock_output.stdout = mock_diff
        mock_run.return_value = mock_output

        diff = get_git_diff()
        assert diff == mock_diff


def test_get_git_diff_error():
    """Test get_git_diff when git command fails."""
    with patch("subprocess.run", side_effect=OSError("git not found")):
        diff = get_git_diff()
        assert diff == ""


def test_get_git_info_integration():
    """Integration test for get_git_info with actual git repo."""
    # Only run if we're in a git repo
    if not os.path.exists(".git"):
        pytest.skip("Not in a git repository")

    commit, branch = get_git_info()

    # Basic validation of output
    assert len(commit) == 40  # SHA-1 hash length
    assert branch != ""  # Should have a branch name


def test_get_git_diff_integration():
    """Integration test for get_git_diff with actual git repo."""
    # Only run if we're in a git repo
    if not os.path.exists(".git"):
        pytest.skip("Not in a git repository")

    diff = get_git_diff()

    # Basic validation of output
    assert isinstance(diff, str)  # Should return a string
    # If there are no changes, diff will be empty
    # If there are changes, it should start with "diff --git"
    assert diff == "" or diff.startswith("diff --git")
