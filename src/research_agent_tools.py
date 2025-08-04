from abc import ABC, abstractmethod
from datetime import datetime
import os
import logging

from research_agent_data_types import ActionResult

logger = logging.getLogger(__name__)


class Tool(ABC):
    """
    Abstract base class for all agent tools.

    This class enforces that all tools implement the 'execute' method,
    ensuring a consistent interface for the agent to use.
    """
    @abstractmethod
    def execute(self, **kwargs) -> ActionResult:
        """Execute the tool's main function.

        Subclasses must implement this method.
        """
        pass


class ReadFileTool(Tool):
    """
    Tool for reading files
    """
    
    def execute(self, file_path: str) -> ActionResult:
        """Read content from a file"""
        try:
            start_time = datetime.now()
            
            if not os.path.exists(file_path):
                return ActionResult(
                    action=f"read_file({file_path})",
                    success=False,
                    output=None,
                    error_message=f"File not found: {file_path}"
                )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Successfully read file: {file_path} ({len(content)} characters)")
            
            return ActionResult(
                action=f"read_file({file_path})",
                success=True,
                output=content,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error reading file {file_path}: {str(e)}")
            
            return ActionResult(
                action=f"read_file({file_path})",
                success=False,
                output=None,
                error_message=str(e),
                execution_time=execution_time
            )


class WriteFileTool(Tool):
    """Tool for writing to new files"""
    
    def execute(self, file_path: str, content: str, overwrite: bool = False) -> ActionResult:
        """Write content to a new file"""
        try:
            start_time = datetime.now()
            
            if os.path.exists(file_path) and not overwrite:
                return ActionResult(
                    action=f"write_file({file_path})",
                    success=False,
                    output=None,
                    error_message=f"File already exists: {file_path}. Use overwrite=True to replace."
                )
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Successfully wrote file: {file_path} ({len(content)} characters)")
            
            return ActionResult(
                action=f"write_file({file_path})",
                success=True,
                output=f"File written successfully: {file_path}",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error writing file {file_path}: {str(e)}")
            
            return ActionResult(
                action=f"write_file({file_path})",
                success=False,
                output=None,
                error_message=str(e),
                execution_time=execution_time
            )