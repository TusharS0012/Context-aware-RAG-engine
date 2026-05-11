import time
import json
import logging
from typing import List, Dict
from src.core.engine import RAGEngine

logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    def __init__(self, engine: RAGEngine):
        self.engine = engine

    async def compare_strategies(self, query: str) -> Dict:
        """Runs both strategies for a query and records metrics."""
        
        # --- Strategy A ---
        start_a = time.perf_counter()
        results_a = await self.engine.strategy_a_direct_search(query)
        latency_a = (time.perf_counter() - start_a) * 1000

        # --- Strategy B ---
        start_b = time.perf_counter()
        results_b = await self.engine.strategy_b_hyde_search(query)
        latency_b = (time.perf_counter() - start_b) * 1000

        return {
            "query": query,
            "strategy_a": {
                "results": [r['content'][:200] + "..." for r in results_a],
                "avg_score": sum(r['score'] for r in results_a) / len(results_a) if results_a else 0,
                "latency_ms": latency_a
            },
            "strategy_b": {
                "results": [r['content'][:200] + "..." for r in results_b],
                "avg_score": sum(r['score'] for r in results_b) / len(results_b) if results_b else 0,
                "latency_ms": latency_b
            }
        }

    def generate_markdown_report(self, all_metrics: List[Dict]):
        """Formats the metrics into the retrieval_benchmark.md file."""
        report = "# RAG Retrieval Benchmark: Strategy A vs Strategy B\n\n"
        report += "## Executive Summary\n"
        report += "- **Strategy A**: Traditional Embedding Similarity\n"
        report += "- **Strategy B**: AI-Enhanced Retrieval (HyDE Expansion)\n\n"
        
        for m in all_metrics:
            report += f"### Query: \"{m['query']}\"\n"
            report += "| Metric | Strategy A (Raw) | Strategy B (AI-Enhanced) |\n"
            report += "| :--- | :--- | :--- |\n"
            report += f"| **Latency** | {m['strategy_a']['latency_ms']:.2f}ms | {m['strategy_b']['latency_ms']:.2f}ms |\n"
            report += f"| **Avg. Similarity Score** | {m['strategy_a']['avg_score']:.4f} | {m['strategy_b']['avg_score']:.4f} |\n\n"
            
            report += "**Top Result A:**\n> " + m['strategy_a']['results'][0] + "\n\n"
            report += "**Top Result B:**\n> " + m['strategy_b']['results'][0] + "\n\n"
            report += "---\n"
            
        with open("retrieval_benchmark.md", "w") as f:
            f.write(report)
        logger.info("✅ Benchmark report generated: retrieval_benchmark.md")