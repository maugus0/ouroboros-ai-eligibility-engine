"""Seed test research embeddings for development and testing."""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from app.config import settings  # pylint: disable=wrong-import-position
from app.repositories.db_pool import (  # pylint: disable=wrong-import-position
    DatabasePoolConfig,
    close_pool,
    create_pool,
)
from app.services.embedding_service import EmbeddingService  # pylint: disable=wrong-import-position

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

SAMPLE_PROGRAM_RESEARCH_FOCUS = [
    {
        "program_id": "10000000-0000-0000-0000-000000000001",
        "text": "Natural language processing, multilingual information retrieval, and human-centered AI",
    },
    {
        "program_id": "10000000-0000-0000-0000-000000000002",
        "text": "Robotics, reinforcement learning, and embodied AI systems",
    },
]


async def seed_embeddings():
    # Initialize database pool
    pool_config = DatabasePoolConfig(
        host=settings.get_db_host(),
        port=settings.get_db_port(),
        database=settings.get_db_name(),
        user=settings.get_db_user(),
        password=settings.get_db_password(),
        min_size=settings.DB_POOL_MIN_SIZE,
        max_size=settings.DB_POOL_MAX_SIZE,
        timeout=settings.DB_CONNECTION_TIMEOUT,
    )
    await create_pool(pool_config)
    print("Database pool initialized.")

    service = EmbeddingService()

    for sample in SAMPLE_RESEARCH_INTERESTS:
        print(f"Generating embedding for profile: {sample['student_profile_id']}")
        try:
            await service.store_student_embedding(
                student_profile_id=sample["student_profile_id"],
                research_interest_text=sample["text"],
            )
            print(f"  \u2713 Stored embedding for {sample['student_profile_id']}")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"  \u2717 Failed: {exc}")

    for sample in SAMPLE_PROGRAM_RESEARCH_FOCUS:
        print(f"Generating embedding for program: {sample['program_id']}")
        try:
            await service.store_program_embedding(
                program_id=sample["program_id"],
                research_focus_text=sample["text"],
            )
            print(f"  \u2713 Stored embedding for {sample['program_id']}")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"  \u2717 Failed: {exc}")

    # Close database pool
    await close_pool()
    print("\nSeeding complete.")


if __name__ == "__main__":
    asyncio.run(seed_embeddings())
