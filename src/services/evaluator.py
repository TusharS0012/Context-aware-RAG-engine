import time
import logging
from typing import List, Dict
from src.core.engine import RAGEngine

logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    def __init__(self, engine: RAGEngine):
        self.engine = engine

    async def compare_strategies(self, query: str) -> dict:
        """Runs both strategies and returns a consistent nested dictionary."""
        
        # 1. Execute Strategy A
        start_a = time.time()
        results_a = await self.engine.query(query, strategy="A")
        latency_a = (time.time() - start_a) * 1000
        
        # 2. Execute Strategy B
        start_b = time.time()
        results_b = await self.engine.query(query, strategy="B")
        latency_b = (time.time() - start_b) * 1000
        
        # Helper to safely get top text
        def get_text(res):
            return res[0].get('text', "No text found") if res else "N/A"

        # Helper to safely get avg score
        def get_score(res):
            if not res: return 0.0
            return sum(r.get('score', 0) for r in res) / len(res)

        # THIS IS THE STRUCTURE main.py EXPECTS
        return {
            "query": query,
            "strategy_a": {
                "latency_ms": latency_a,
                "avg_score": get_score(results_a),
                "results": [get_text(results_a)]
            },
            "strategy_b": {
                "latency_ms": latency_b,
                "avg_score": get_score(results_b),
                "results": [get_text(results_b)]
            }
        }

    def generate_markdown_report(self, all_metrics: List[Dict]):
        """Generates the benchmark report from the session metrics."""
        report = "# RAG Retrieval Benchmark: Strategy A vs Strategy B\n\n"
        report += "## Executive Summary\n"
        report += "- **Strategy A**: Traditional Embedding Similarity\n"
        report += "- **Strategy B**: AI-Enhanced Retrieval (HyDE Expansion)\n\n"
        
        for m in all_metrics:
            # Ensure we are accessing the nested structure correctly
            query_text = m.get('query', 'Unknown Query')
            report += f"### Query: \"{query_text}\"\n"
            report += "| Metric | Strategy A (Raw) | Strategy B (AI-Enhanced) |\n"
            report += "| :--- | :--- | :--- |\n"
            report += f"| **Latency** | {m['strategy_a']['latency_ms']:.2f}ms | {m['strategy_b']['latency_ms']:.2f}ms |\n"
            report += f"| **Avg. Similarity Score** | {m['strategy_a']['avg_score']:.4f} | {m['strategy_b']['avg_score']:.4f} |\n\n"
            
            report += "**Top Result A:**\n> " + m['strategy_a']['results'][0] + "\n\n"
            report += "**Top Result B:**\n> " + m['strategy_b']['results'][0] + "\n\n"
            report += "---\n"
            
        with open("retrieval_benchmark.md", "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("✅ Benchmark report generated: retrieval_benchmark.md")