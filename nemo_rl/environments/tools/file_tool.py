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

class FileTool:
    def __init__(self, max_file_size: int = 1024 * 1024):  # 1MB default
        self.max_file_size = max_file_size
    
    def _validate_path(self, path: str, working_dir: str) -> tuple[bool, str, str]:
        if not os.path.exists(working_dir):
            try:
                os.makedirs(working_dir, exist_ok=True)
            except Exception as e:
                return False, "", f"Failed to create working directory '{working_dir}': {str(e)}"
        
        if os.path.isabs(path):
            full_path = os.path.abspath(path)
        else:
            full_path = os.path.abspath(os.path.join(working_dir, path))
        
        working_dir_abs = os.path.abspath(working_dir)
        if not full_path.startswith(working_dir_abs + os.sep) and full_path != working_dir_abs:
            return False, full_path, f"Path '{path}' is outside the working directory"
        
        return True, full_path, ""
    
    def create(self, path: str, content: str, working_dir: str) -> str:
        is_valid, full_path, error_msg = self._validate_path(path, working_dir)
        if not is_valid:
            return f"Error: {error_msg}"
        
        if os.path.exists(full_path):
            return f"Error: File '{path}' already exists. Use file_edit to modify it."
        
        if len(content.encode('utf-8')) > self.max_file_size:
            return f"Error: File content exceeds maximum size of {self.max_file_size} bytes"
        
        try:
            parent_dir = os.path.dirname(full_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"File created successfully at '{path}'"
        except Exception as e:
            return f"Error creating file: {str(e)}"
    
    def edit(self, path: str, old_content: str, new_content: str, working_dir: str) -> str:
        is_valid, full_path, error_msg = self._validate_path(path, working_dir)
        if not is_valid:
            return f"Error: {error_msg}"
        
        if not os.path.exists(full_path):
            try:
                parent_dir = os.path.dirname(full_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write("")
            except Exception as e:
                return f"Error creating file '{path}': {str(e)}"
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                current_content = f.read()
            
            if old_content == "":
                updated_content = current_content + new_content
            elif old_content in current_content:
                updated_content = current_content.replace(old_content, new_content)
            else:
                return f"Error: Content to replace not found in file '{path}'"
            
            if len(updated_content.encode('utf-8')) > self.max_file_size:
                return f"Error: Modified file would exceed maximum size of {self.max_file_size} bytes"
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            return f"File '{path}' updated successfully"
            
        except Exception as e:
            return f"Error editing file '{path}': {str(e)}"
    

    
    def read(self, path: str, working_dir: str) -> str:
        is_valid, full_path, error_msg = self._validate_path(path, working_dir)
        if not is_valid:
            return f"Error: {error_msg}"
        
        if not os.path.exists(full_path):
            return f"Error: File '{path}' does not exist"
        
        try:
            file_size = os.path.getsize(full_path)
            if file_size > self.max_file_size:
                return f"Error: File size ({file_size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)"
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return f"Contents of '{path}':\n{content}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def delete(self, path: str, working_dir: str) -> str:
        is_valid, full_path, error_msg = self._validate_path(path, working_dir)
        if not is_valid:
            return f"Error: {error_msg}"

        if not os.path.exists(full_path):
            return f"Error: File '{path}' does not exist"
        
        if os.path.isdir(full_path):
            return f"Error: '{path}' is a directory. Only files can be deleted."
        
        try:
            os.remove(full_path)
            return f"File '{path}' deleted successfully"
        except Exception as e:
            return f"Error deleting file: {str(e)}"
    
    def list_directory(self, path: str, working_dir: str) -> str:
        is_valid, full_path, error_msg = self._validate_path(path, working_dir)
        if not is_valid:
            return f"Error: {error_msg}"
        
        if not os.path.exists(full_path):
            if full_path == os.path.abspath(working_dir):
                try:
                    os.makedirs(full_path, exist_ok=True)
                except Exception as e:
                    return f"Error: Failed to create directory '{path}': {str(e)}"
            else:
                return f"Error: Directory '{path}' does not exist"
        
        if not os.path.isdir(full_path):
            return f"Error: '{path}' is not a directory"
        
        try:
            items = []
            for item in sorted(os.listdir(full_path)):
                item_path = os.path.join(full_path, item)
                if os.path.isdir(item_path):
                    items.append(f"[DIR]  {item}/")
                else:
                    size = os.path.getsize(item_path)
                    items.append(f"[FILE] {item} ({size} bytes)")
            
            if not items:
                return f"Directory '{path}' is empty"
            
            return f"Contents of '{path}':\n" + "\n".join(items)
        except Exception as e:
            return f"Error listing directory: {str(e)}" 

