"""
Mock Neo4j Client for testing retrieval pipeline without a real Neo4j instance.
Simulates graph-based retrieval and SIMILAR_TO relationship traversal.
"""

import random
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase


class MockNeo4jClient:
    """Mock Neo4j client that simulates graph database behavior for testing."""

    def __init__(
        self, uri: str = "mock://localhost:7687", auth: tuple = ("neo4j", "password")
    ):
        self.driver = True  # Simulate connected driver
        self._mock_data = {}
        self._similar_relationships = {}
        print(f"MockNeo4j connected (simulating {uri})")
        self._init_constraints()

    def close(self):
        """Simulate closing connection."""
        self.driver = None
        print("MockNeo4j connection closed")

    def _init_constraints(self):
        """Simulate constraint initialization."""
        self._mock_data = {"images": {}, "concepts": {}, "texts": {}}
        self._similar_relationships = {}
        print("MockNeo4j constraints initialized")

    def add_image_node(self, image_id: str):
        """Add an image node to the mock graph."""
        if "images" not in self._mock_data:
            self._mock_data["images"] = {}
        self._mock_data["images"][image_id] = {
            "id": image_id,
            "concepts": {"HAS": [], "NOT_HAS": []},
        }

    def add_concepts(
        self, image_id: str, pos_concepts: List[str], neg_concepts: List[str]
    ):
        """Add concepts to an image node."""
        if image_id in self._mock_data["images"]:
            img = self._mock_data["images"][image_id]
            img["concepts"]["HAS"].extend(pos_concepts)
            img["concepts"]["NOT_HAS"].extend(neg_concepts)

            for concept in pos_concepts:
                if concept not in self._mock_data["concepts"]:
                    self._mock_data["concepts"][concept] = []
                self._mock_data["concepts"][concept].append(image_id)

    def add_similar_relationship(
        self, img_id1: str, img_id2: str, similarity_score: float
    ):
        """Add SIMILAR_TO relationship between two images."""
        if img_id1 not in self._similar_relationships:
            self._similar_relationships[img_id1] = []
        self._similar_relationships[img_id1].append(
            {"image_id": img_id2, "score": similarity_score}
        )

    def get_similar_images(self, img_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get images similar to the given image via SIMILAR_TO relationships."""
        results = self._similar_relationships.get(img_id, [])
        return results[:limit]

    def search_graph(
        self, pos_entities: List[str], neg_entities: List[str], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search graph for images matching the given entities.
        Simulates Neo4j concept-based retrieval.
        """
        candidates = {}

        for entity in pos_entities:
            entity_lower = entity.lower()
            for concept, images in self._mock_data["concepts"].items():
                concept_lower = concept.lower()
                # Simple substring matching for mock
                if entity_lower in concept_lower or concept_lower in entity_lower:
                    for img_id in images:
                        if img_id not in candidates:
                            candidates[img_id] = 0
                        candidates[img_id] += 1.0  # Match score

        # Sort by score and return top results
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [
            {"image_id": img_id, "score": score}
            for img_id, score in sorted_candidates[:limit]
        ]

    def search_mixed_modality(
        self,
        pos_entities: List[str],
        neg_entities: List[str],
        limit_img: int = 5,
        limit_text: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Simulate mixed modality search."""
        images = self.search_graph(pos_entities, neg_entities, limit_img)
        return {"images": images, "texts": []}

    def add_text_node(self, text_id: str, content: str, metadata: str = "{}"):
        """Add a text node."""
        if "texts" not in self._mock_data:
            self._mock_data["texts"] = {}
        self._mock_data["texts"][text_id] = {
            "id": text_id,
            "content": content,
            "metadata": metadata,
        }

    def add_text_concepts(self, text_id: str, concepts: List[str]):
        """Add concepts to a text node."""
        # Mock implementation - just store the concepts
        pass

    def populate_mock_data(self, sample_data: Dict):
        """Populate mock database with sample data for testing."""
        for img_id, concepts in sample_data.items():
            self.add_image_node(img_id)
            pos = concepts.get("positive", [])
            neg = concepts.get("negative", [])
            self.add_concepts(img_id, pos, neg)


# Example usage and test
if __name__ == "__main__":
    # Create mock client
    mock_neo4j = MockNeo4jClient()

    # Add some sample data
    mock_neo4j.add_image_node("img_001")
    mock_neo4j.add_image_node("img_002")
    mock_neo4j.add_image_node("img_003")

    mock_neo4j.add_concepts("img_001", ["dog", "terrier", "black"], ["cat"])
    mock_neo4j.add_concepts("img_002", ["dog", "terrier", "brown"], ["cat"])
    mock_neo4j.add_concepts("img_003", ["cat", "orange"], ["dog"])

    # Add similar relationships
    mock_neo4j.add_similar_relationship("img_001", "img_002", 0.85)
    mock_neo4j.add_similar_relationship("img_002", "img_001", 0.85)

    # Test search
    print("\nSearching for 'terrier'...")
    results = mock_neo4j.search_graph(["terrier"], [], limit=5)
    print(f"Results: {results}")

    # Test similar images
    print("\nGetting similar images for img_001...")
    similar = mock_neo4j.get_similar_images("img_001", limit=3)
    print(f"Similar: {similar}")

    mock_neo4j.close()
