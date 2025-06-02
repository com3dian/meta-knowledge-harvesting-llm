from typing import List, Set, Dict
from dataclasses import dataclass
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

console = Console()

@dataclass
class HighlightResult:
    highlighted_text: str
    not_found_substrings: List[str]
    not_found_keywords: List[str]

class TextHighlighter:
    def __init__(self):
        self.console = Console()

    def highlight_text(self, text: str, substrings: List[str], keywords: List[str]) -> HighlightResult:
        """
        Highlight substrings and keywords in the text.
        
        Args:
            text: The main text to highlight
            substrings: List of substrings to highlight in yellow
            keywords: List of keywords to highlight in red (only within substrings)
            
        Returns:
            HighlightResult containing the highlighted text and lists of not found items
        """
        # Track which substrings and keywords were found
        found_substrings = set()
        found_keywords = set()
        
        # First, find all substring positions
        substring_positions = []
        for substring in substrings:
            start = 0
            while True:
                pos = text.lower().find(substring.lower(), start)
                if pos == -1:
                    break
                substring_positions.append((pos, pos + len(substring), substring))
                found_substrings.add(substring)
                start = pos + 1

        # Sort positions by start index
        substring_positions.sort(key=lambda x: x[0])

        # Find all keyword positions within substrings
        keyword_positions = []
        for start, end, substring in substring_positions:
            substring_text = text[start:end]
            for keyword in keywords:
                k_start = 0
                while True:
                    k_pos = substring_text.lower().find(keyword.lower(), k_start)
                    if k_pos == -1:
                        break
                    keyword_positions.append((start + k_pos, start + k_pos + len(keyword), keyword))
                    found_keywords.add(keyword)
                    k_start = k_pos + 1

        # Sort keyword positions
        keyword_positions.sort(key=lambda x: x[0])

        # Build the highlighted text using rich.Text
        rich_text = Text()
        last_pos = 0

        # Process all positions (both substrings and keywords)
        all_positions = substring_positions + keyword_positions
        all_positions.sort(key=lambda x: x[0])

        for start, end, _ in all_positions:
            # Add text before the highlight
            if start > last_pos:
                rich_text.append(text[last_pos:start])
            
            # Add the highlighted text
            highlighted_text = text[start:end]
            if (start, end) in [(s[0], s[1]) for s in keyword_positions]:
                rich_text.append(highlighted_text, style="bold red")
            else:
                rich_text.append(highlighted_text, style="bold yellow")
            
            last_pos = end

        # Add any remaining text
        if last_pos < len(text):
            rich_text.append(text[last_pos:])

        # Find not found items
        not_found_substrings = [s for s in substrings if s not in found_substrings]
        not_found_keywords = [k for k in keywords if k not in found_keywords]

        return HighlightResult(
            highlighted_text=str(rich_text),
            not_found_substrings=not_found_substrings,
            not_found_keywords=not_found_keywords
        )

    def print_result(self, result: HighlightResult):
        """Pretty print the highlighting results using rich."""
        # Create a panel for the highlighted text
        text_panel = Panel(
            Text(result.highlighted_text),
            title="Highlighted Text",
            border_style="blue"
        )
        self.console.print(text_panel)

        # Create a table for not found items
        if result.not_found_substrings or result.not_found_keywords:
            table = Table(title="Not Found Items", border_style="red")
            table.add_column("Type", style="cyan")
            table.add_column("Items", style="yellow")

            if result.not_found_substrings:
                table.add_row(
                    "Substrings",
                    ", ".join(result.not_found_substrings)
                )
            if result.not_found_keywords:
                table.add_row(
                    "Keywords",
                    ", ".join(result.not_found_keywords)
                )
            
            self.console.print(table)

def main():
    # Example usage
    highlighter = TextHighlighter()
    
    text = "The quick brown fox jumps over the lazy dog. The fox is quick and brown."
    substrings = ["quick brown", "lazy dog", "not found"]
    keywords = ["fox", "quick", "missing"]
    
    result = highlighter.highlight_text(text, substrings, keywords)
    highlighter.print_result(result)

if __name__ == "__main__":
    main() 