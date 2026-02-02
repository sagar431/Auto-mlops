#!/usr/bin/env python
"""Script to run the image classification example with Auto-MLOps agent."""

import asyncio
import sys
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


async def run_example(query: str = None, project_path: str = None, threshold: float = 0.85):
    """Run the Auto-MLOps agent on the image classification example."""
    from agent import AgentLoop

    if project_path is None:
        project_path = str(Path(__file__).parent / "project")

    if query is None:
        query = "Set up an MLOps pipeline for image classification with training and evaluation"

    print(f"Project path: {project_path}")
    print(f"Query: {query}")
    print(f"Accuracy threshold: {threshold}")
    print("-" * 50)

    agent = AgentLoop()
    result = await agent.run(
        query=query,
        project_path=project_path,
        accuracy_threshold=threshold,
    )

    print("\n" + "=" * 50)
    print("Agent Result:")
    print("=" * 50)
    print(result)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Auto-MLOps on image classification example")
    parser.add_argument(
        "--query",
        type=str,
        default="Set up an MLOps pipeline for image classification",
        help="Query to send to the agent",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Path to project directory",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Accuracy threshold",
    )

    args = parser.parse_args()

    asyncio.run(
        run_example(
            query=args.query,
            project_path=args.project,
            threshold=args.threshold,
        )
    )


if __name__ == "__main__":
    main()
