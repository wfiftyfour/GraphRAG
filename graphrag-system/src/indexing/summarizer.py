"""Generate community summaries using LLM."""

import json
import requests
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd


class CommunitySummarizer:
    """Generate summaries for communities."""

    def __init__(self, output_dir: str = "data/output/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model = "llama3.1:8b"

    def summarize_community(self, community_id: int, members: List[str],
                           graph, prompt_template: str = None) -> Dict[str, Any]:
        """Generate summary for a single community."""
        # Get entity descriptions and relationships
        entity_info = []
        for member in members:
            if graph.has_node(member):
                attrs = graph.nodes[member]
                entity_info.append(f"- {member} ({attrs.get('type', 'UNKNOWN')}): {attrs.get('description', '')}")

        # Get internal relationships
        relationships = []
        for member in members:
            for neighbor in graph.neighbors(member):
                if neighbor in members:
                    edge_data = graph.edges[member, neighbor]
                    relationships.append(
                        f"- {member} --[{edge_data.get('relationship', '')}]--> {neighbor}"
                    )

        prompt = prompt_template or self._default_prompt()
        full_prompt = prompt.format(
            entities='\n'.join(entity_info),
            relationships='\n'.join(relationships[:20])  # Limit relationships
        )

        summary = self._call_llm(full_prompt)

        return {
            'community_id': community_id,
            'title': self._generate_title(members),
            'summary': summary,
            'num_entities': len(members),
            'entities': members,
            'rank': len(members)  # Simple ranking by size
        }

    def summarize_all(self, communities: Dict[int, List[str]],
                      graph, prompt_template: str = None) -> List[Dict[str, Any]]:
        """Generate summaries for all communities."""
        reports = []

        for community_id, members in communities.items():
            print(f"Summarizing community {community_id} ({len(members)} entities)...")
            report = self.summarize_community(
                community_id, members, graph, prompt_template
            )
            reports.append(report)

        return reports

    def save(self, reports: List[Dict[str, Any]]):
        """Save community reports."""
        # Save as parquet
        df_data = []
        for report in reports:
            df_data.append({
                'community_id': report['community_id'],
                'title': report['title'],
                'summary': report['summary'],
                'num_entities': report['num_entities'],
                'rank': report['rank']
            })

        df = pd.DataFrame(df_data)
        df.to_parquet(self.output_dir / "community_reports.parquet", index=False)

        print(f"Saved {len(reports)} community reports")

    def load(self) -> List[Dict[str, Any]]:
        """Load community reports."""
        df = pd.read_parquet(self.output_dir / "community_reports.parquet")
        return df.to_dict('records')

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama LLM."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.3}
        }

        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            return response.json()['message']['content']
        except Exception as e:
            return f"Summary generation failed: {e}"

    def _generate_title(self, members: List[str]) -> str:
        """Generate a title for the community."""
        if len(members) <= 3:
            return ', '.join(members)
        return f"{members[0]}, {members[1]}, and {len(members)-2} others"

    def _default_prompt(self) -> str:
        return """Create a comprehensive summary of this community of related entities.

Entities:
{entities}

Key Relationships:
{relationships}

Write a 2-3 paragraph summary that:
1. Identifies the main theme or topic of this community
2. Describes the key entities and their roles
3. Explains the important relationships between entities

Summary:"""
