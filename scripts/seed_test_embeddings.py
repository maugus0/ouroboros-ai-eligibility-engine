"""Seed test research embeddings for development and testing."""

import asyncio
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

from app.config import settings
from app.services.embedding_service import EmbeddingService

SAMPLE_RESEARCH_INTERESTS = [
    {
        "student_profile_id": "00000000-0000-0000-0000-000000000001",
        "text": "Natural language processing, transformer architectures, and multilingual sentiment analysis",
    },
    {
        "student_profile_id": "00000000-0000-0000-0000-000000000002",
        "text": "Computer vision applications in autonomous vehicles and LiDAR point cloud processing",
    },
    {
        "student_profile_id": "00000000-0000-0000-0000-000000000003",
        "text": "Reinforcement learning for robotics control and multi-agent coordination systems",
    },
    {
        "student_profile_id": "00000000-0000-0000-0000-000000000004",
        "text": "Computational biology, protein structure prediction, and drug discovery using graph neural networks",
    },
    {
        "student_profile_id": "00000000-0000-0000-0000-000000000005",
        "text": "Federated learning for privacy-preserving healthcare data analysis and differential privacy",
    },
]


async def seed_embeddings():
    service = EmbeddingService()

    for sample in SAMPLE_RESEARCH_INTERESTS:
        print(f"Generating embedding for profile: {sample['student_profile_id']}")
        try:
            await service.store_embedding(
                student_profile_id=sample["student_profile_id"],
                research_interest_text=sample["text"],
            )
            print(f"  \u2713 Stored embedding for {sample['student_profile_id']}")
        except Exception as exc:
            print(f"  \u2717 Failed: {exc}")

    print("\nSeeding complete.")


if __name__ == "__main__":
    asyncio.run(seed_embeddings())
