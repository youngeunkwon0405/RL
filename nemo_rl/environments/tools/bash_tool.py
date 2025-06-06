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
import subprocess
import threading
from typing import Optional


class BashTool:
    """Tool for executing bash commands with safety constraints."""
    
    BLACKLISTED_COMMANDS = {
        "rm -rf /",
        "dd",
        "mkfs",
        "shutdown",
        "reboot",
        "systemctl",
        "service",
        "kill",
        "pkill",
        "killall",
        "sudo",
        "su",
        "passwd",
        "useradd",
        "userdel",
        "groupadd",
        "groupdel",
        "chown",
        "chmod 777",
        "iptables",
        "nc",
        "netcat",
        "wget",
        "curl",
        "ssh",
        "scp",
        "rsync",
    }
    
    def __init__(
        self,
        timeout: int = 30,
        memory_limit: int = 512,  # MB
        cpu_limit: float = 1.0,  # CPU cores
        enable_network: bool = False,
    ):
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.enable_network = enable_network
    
    def _is_command_safe(self, command: str) -> bool:
        command_lower = command.lower().strip()
        command_words = command_lower.split()
        
        for blacklisted in self.BLACKLISTED_COMMANDS:
            if ' ' in blacklisted:
                if blacklisted in command_lower:
                    return False
            else:
                if blacklisted in command_words:
                    return False
        
        dangerous_patterns = [
            ">/dev/",
            "&>/dev/",
            "2>/dev/",
            "|sudo",
            "|su",
            "$(sudo",
            "`sudo",
            "$(su",
            "`su",
        ]
        
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                return False
        
        if not self.enable_network:
            network_commands = ["ping", "telnet", "ftp", "sftp", "dig", "nslookup", "traceroute"]
            for net_cmd in network_commands:
                if net_cmd in command_words:
                    return False
        
        return True
    
    def _run_with_timeout(self, proc: subprocess.Popen, timeout: int) -> tuple[str, str, bool]:
        stdout_data = []
        stderr_data = []
        timed_out = False
        
        def target():
            stdout, stderr = proc.communicate()
            stdout_data.append(stdout)
            stderr_data.append(stderr)
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            proc.terminate()
            thread.join()
            timed_out = True
        
        stdout = stdout_data[0] if stdout_data else b""
        stderr = stderr_data[0] if stderr_data else b""
        
        return stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace"), timed_out
    
    def execute(self, command: str, working_dir: str) -> str:
        if not os.path.exists(working_dir):
            try:
                os.makedirs(working_dir, exist_ok=True)
            except Exception as e:
                return f"Error: Failed to create working directory '{working_dir}': {str(e)}"
        
        if not os.path.isdir(working_dir):
            return f"Error: '{working_dir}' is not a directory"
        
        if not self._is_command_safe(command):
            return f"Error: Command '{command}' is not allowed for safety reasons"
        
        env = os.environ.copy()
        
        resource_limits = []
        if self.memory_limit:
            memory_kb = self.memory_limit * 1024
            resource_limits.append(f"ulimit -v {memory_kb}")
        
        if self.cpu_limit:
            cpu_seconds = int(self.timeout * self.cpu_limit)
            resource_limits.append(f"ulimit -t {cpu_seconds}")
        
        if resource_limits:
            full_command = f"{'; '.join(resource_limits)}; {command}"
        else:
            full_command = command
        
        try:
            proc = subprocess.Popen(
                full_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
                env=env,
                preexec_fn=os.setsid if os.name != 'nt' else None,
            )
            
            stdout, stderr, timed_out = self._run_with_timeout(proc, self.timeout)
            
            if timed_out:
                return f"Error: Command timed out after {self.timeout} seconds"
            
            output = stdout
            if stderr:
                output += f"\n[stderr]\n{stderr}"
            
            if proc.returncode != 0:
                output = f"Command exited with code {proc.returncode}\n{output}"
            
            max_output_length = 10000
            if len(output) > max_output_length:
                output = output[:max_output_length] + f"\n... (truncated, output too long)"
            
            return output.strip() if output else "(no output)"
            
        except Exception as e:
            return f"Error executing command: {str(e)}" 

