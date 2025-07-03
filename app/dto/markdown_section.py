from dataclasses import dataclass

@dataclass
class MarkdownSection:
    title: str
    content: str
    
    def __len__(self) -> int:
        return len(self.content)