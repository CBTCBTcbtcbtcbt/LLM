"""Paper search tool for Agent integration with multimodal support."""
import os
from pathlib import Path
from typing import Dict, Any, Union, Optional

# Tool declaration for function calling
search_paper_declaration = {
    "name": "search_paper",
    "description": """Search and retrieve paper information. This tool has two modes:
    1. 'abstract' mode: Returns a summary/abstract of all available papers from abstract.txt
    2. 'content' mode: Returns the full content of a specific paper PDF file for detailed analysis
    
    Use 'abstract' mode when user wants to know what papers are available or needs a quick overview.
    Use 'content' mode when user wants to read the detailed content of a specific paper.""",
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["abstract", "content"],
                "description": "The search mode. Use 'abstract' for paper summaries, 'content' for full paper PDF."
            },
            "filename": {
                "type": "string",
                "description": "The PDF filename to retrieve (only required when mode is 'content'). Example: '1-s2.0-S0378383922000412-main.pdf'"
            }
        },
        "required": ["mode"]
    }
}


def search_paper(mode: str, filename: Optional[str] = None, base_path: Optional[Union[str, Path]] = None) -> Union[str, Dict[str, Any]]:
    """
    Search and retrieve paper information.
    
    Args:
        mode: 'abstract' for paper summaries, 'content' for full PDF content.
        filename: PDF filename (required when mode='content').
        base_path: Base directory path (defaults to current working directory).
    
    Returns:
        For 'abstract' mode: String content of abstract.txt
        For 'content' mode: Dict with file info for multimodal processing, or error string
    """
    if base_path is None:
        # Default to parent directory of LLM folder (the paper processer root)
        resolved_path = Path(__file__).parent.parent
    else:
        resolved_path = Path(base_path)
    
    if mode == "abstract":
        # Return the abstract.txt content
        abstract_path = resolved_path / "abstract.txt"
        
        if not abstract_path.exists():
            return f"Error: abstract.txt not found at {abstract_path}"
        
        try:
            with open(abstract_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading abstract.txt: {str(e)}"
    
    elif mode == "content":
        if not filename:
            return "Error: 'filename' parameter is required when mode is 'content'"
        
        # Security check: prevent path traversal
        safe_filename = Path(filename).name
        if safe_filename != filename:
            return f"Error: Invalid filename format. Please provide only the filename without path."
        
        input_folder = resolved_path / "input"
        file_path = input_folder / safe_filename
        
        if not file_path.exists():
            # Try to find a similar file
            available_files = list(input_folder.glob("*.pdf"))
            available_names = [f.name for f in available_files]
            
            # Check for partial match
            matches = [name for name in available_names if filename.lower() in name.lower()]
            
            if matches:
                suggestion = f"Did you mean one of these files?\n" + "\n".join(f"- {m}" for m in matches[:5])
            else:
                suggestion = f"Available PDF files:\n" + "\n".join(f"- {name}" for name in available_names[:10])
                if len(available_names) > 10:
                    suggestion += f"\n... and {len(available_names) - 10} more files"
            
            return f"Error: File '{safe_filename}' not found in input folder.\n{suggestion}"
        
        # Return a special marker for multimodal file processing
        # This will be detected by the LLM client and converted to a proper Part
        return {
            "__multimodal_file__": True,
            "file_path": str(file_path),
            "mime_type": "application/pdf",
            "filename": safe_filename,
            "description": f"PDF content of paper: {safe_filename}"
        }
    
    else:
        return f"Error: Unknown mode '{mode}'. Use 'abstract' or 'content'."


def get_paper_tool_handler(base_path: str = None):
    """
    Create a paper tool handler with a specific base path.
    
    Args:
        base_path: Base directory path for finding papers.
    
    Returns:
        A function that can be registered as a tool handler.
    """
    def handler(mode: str, filename: str = None) -> Union[str, Dict[str, Any]]:
        return search_paper(mode=mode, filename=filename, base_path=base_path)
    
    return handler


# For convenience, also export as a simple handler that uses default paths
def default_search_paper_handler(mode: str, filename: str = None) -> Union[str, Dict[str, Any]]:
    """Default handler using the standard directory structure."""
    return search_paper(mode=mode, filename=filename)
