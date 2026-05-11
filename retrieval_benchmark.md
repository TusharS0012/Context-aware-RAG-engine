# RAG Retrieval Benchmark: Strategy A vs Strategy B

## Executive Summary
- **Strategy A**: Traditional Embedding Similarity
- **Strategy B**: AI-Enhanced Retrieval (HyDE Expansion)

### Query: "How does the system handle peak load?"
| Metric | Strategy A (Raw) | Strategy B (AI-Enhanced) |
| :--- | :--- | :--- |
| **Latency** | 540.12ms | 3301.57ms |
| **Avg. Similarity Score** | 0.5806 | 0.6245 |

**Top Result A:**
> Technical documentation fragment 380: This section discusses peak load handling and horizontal scaling including details on redundancy, database sharding, and failover mechanisms for index 380....

**Top Result B:**
> Technical documentation fragment 630: This section discusses peak load handling and horizontal scaling including details on redundancy, database sharding, and failover mechanisms for index 630....

---
### Query: "What are the database sharding strategies?"
| Metric | Strategy A (Raw) | Strategy B (AI-Enhanced) |
| :--- | :--- | :--- |
| **Latency** | 661.52ms | 16422.97ms |
| **Avg. Similarity Score** | 0.5586 | 0.5371 |

**Top Result A:**
> Technical documentation fragment 747: This section discusses general system architecture including details on redundancy, database sharding, and failover mechanisms for index 747....

**Top Result B:**
> Technical documentation fragment 743: This section discusses general system architecture including details on redundancy, database sharding, and failover mechanisms for index 743....

---
### Query: "Explain the redundancy and failover mechanisms."
| Metric | Strategy A (Raw) | Strategy B (AI-Enhanced) |
| :--- | :--- | :--- |
| **Latency** | 614.56ms | 2598.72ms |
| **Avg. Similarity Score** | 0.6462 | 0.6430 |

**Top Result A:**
> Technical documentation fragment 334: This section discusses general system architecture including details on redundancy, database sharding, and failover mechanisms for index 334....

**Top Result B:**
> Technical documentation fragment 334: This section discusses general system architecture including details on redundancy, database sharding, and failover mechanisms for index 334....

---
