"""Quick config validation check."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Steering.config.pipeline import PipelineConfig
c = PipelineConfig.load(sys.argv[1])
print(f"load_vector: {c.load_vector}")
print(f"extractor method: {c.extractor.method}, layer: {c.extractor.layer}")
print(f"steer method: {c.steer.method}, coeff: {c.steer.coeff}")
print(f"chars_tail_transform (steer): {getattr(c.steer, 'chars_tail_transform', 'N/A')}")
print(f"test_dataset: {c.test_dataset}")
