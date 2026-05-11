import asyncio
import logging
from src.services.ingestion import IngestionService
from src.services.evaluator import RetrievalEvaluator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def main():
    # 1. Generate 1,000 Mock Paragraphs for the "Senior" Volume Requirement
    # In a real scenario, you'd load this from a JSON/PDF.
    mock_data = [
        f"Technical documentation fragment {i}: This section discusses "
        f"{'peak load handling and horizontal scaling' if i % 10 == 0 else 'general system architecture'} "
        f"including details on redundancy, database sharding, and failover mechanisms for index {i}."
        for i in range(1000)
    ]

    # 2. Ingestion
    ingestor = IngestionService(batch_size=25)
    await ingestor.run_ingestion(mock_data)

    # 3. Evaluation
    evaluator = RetrievalEvaluator(ingestor.engine)
    test_queries = [
        "How does the system handle peak load?",
        "What are the database sharding strategies?",
        "Explain the redundancy and failover mechanisms."
    ]

    results = []
    print("\n🚀 Starting Benchmarking Tasks...")
    for q in test_queries:
        metric = await evaluator.compare_strategies(q)
        results.append(metric)
    
    # 4. Generate Report
    evaluator.generate_markdown_report(results)

if __name__ == "__main__":
    asyncio.run(main())