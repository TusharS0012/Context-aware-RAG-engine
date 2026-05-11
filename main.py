import asyncio
import os
import logging
from src.core.engine import RAGEngine
from src.services.ingestion import IngestionService
from src.services.evaluator import RetrievalEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

async def main():
    engine = RAGEngine()
    ingestor = IngestionService(engine)
    evaluator = RetrievalEvaluator(engine)

    # 1. Find PDF
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    if not pdf_files:
        print("❌ Error: No PDF file found.")
        return
    
    # 2. Ingest
    await ingestor.run_ingestion(pdf_files[0])

    print(f"\n🚀 RAG System Ready (Context: {pdf_files[0]})")
    session_metrics = []

    # 3. Loop
    while True:
        query = input("\n🔍 Query (or 'exit'): ").strip()
        if query.lower() in ['exit', 'quit', 'q']:
            break
        if not query:
            continue

        try:
            # This now returns the nested dict with 'query', 'strategy_a', and 'strategy_b'
            metric = await evaluator.compare_strategies(query)
            session_metrics.append(metric)
            
            print(f"\n📊 Strategy A | Score: {metric['strategy_a']['avg_score']:.4f} | {metric['strategy_a']['latency_ms']:.1f}ms")
            print(f"🧠 Strategy B | Score: {metric['strategy_b']['avg_score']:.4f} | {metric['strategy_b']['latency_ms']:.1f}ms")
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")

    # 4. Final Report
    if session_metrics:
        evaluator.generate_markdown_report(session_metrics)

if __name__ == "__main__":
    asyncio.run(main())