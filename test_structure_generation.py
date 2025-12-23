import unittest
import numpy as np

from structure_generation import map_czbiohub_and_swift_structures


SIMILARITY_EPS = 1e-8


class MapStructureGenerationTests(unittest.TestCase):
    def test_relationship_map_between_contexts(self):
        result = map_czbiohub_and_swift_structures(seed=123)

        self.assertIn("czbiohub", result)
        self.assertIn("swift_market_making", result)
        self.assertIn("relationship_map", result)

        relationship = result["relationship_map"]
        self.assertEqual(relationship.shape, (2, 2))

        np.testing.assert_allclose(relationship, relationship.T)
        self.assertTrue(np.all(relationship >= 0))
        self.assertTrue(np.all(relationship <= 1.0 + SIMILARITY_EPS))
        self.assertGreater(relationship[0, 1], 0)


if __name__ == "__main__":
    unittest.main()
