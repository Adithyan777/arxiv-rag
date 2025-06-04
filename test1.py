from streamlit_ui.components import get_paper_metadata
from qdrant_client import QdrantClient
from init_vector_stores.papers import init_vector_store
from init_vector_stores.abstracts import AbstractVectorStore
from utils import add_paper_to_json

# add_paper_to_json({
#   "title": "GaRA-SAM: Robustifying Segment Anything Model with Gated-Rank Adaptation",
#   "authors": "['sohyun lee', 'yeho kwon', 'lukas hoyer', 'suha kwak']",
#   "abstract": "Improving robustness of the Segment Anything Model (SAM) to input degradations is critical for its deployment in high-stakes applications such as autonomous driving and robotics. Our approach to this challenge prioritizes three key aspects: first, parameter efficiency to maintain the inherent generalization capability of SAM; second, fine-grained and input-aware robustification to precisely address the input corruption; and third, adherence to standard training protocols for ease of training. To this end, we propose gated-rank adaptation (GaRA). GaRA introduces lightweight adapters into intermediate layers of the frozen SAM, where each adapter dynamically adjusts the effective rank of its weight matrix based on the input by selectively activating (rank-1) components of the matrix using a learned gating module. This adjustment enables fine-grained and input-aware robustification without compromising the generalization capability of SAM. Our model, GaRA-SAM, significantly outperforms prior work on all robust segmentation benchmarks. In particular, it surpasses the previous best IoU score by up to 21.3\\%p on ACDC, a challenging real corrupted image dataset.",
#   "created": "2025-06-03",
#   "updated": "2025-06-03",
#   "id": "2506.02882",
#   "categories": "cs.CV"
# })
# print("Paper added to JSON successfully.")

metadata = {
  "title": "GaRA-SAM: Robustifying Segment Anything Model with Gated-Rank Adaptation",
  "authors": "['sohyun lee', 'yeho kwon', 'lukas hoyer', 'suha kwak']",
  "abstract": "Improving robustness of the Segment Anything Model (SAM) to input degradations is critical for its deployment in high-stakes applications such as autonomous driving and robotics. Our approach to this challenge prioritizes three key aspects: first, parameter efficiency to maintain the inherent generalization capability of SAM; second, fine-grained and input-aware robustification to precisely address the input corruption; and third, adherence to standard training protocols for ease of training. To this end, we propose gated-rank adaptation (GaRA). GaRA introduces lightweight adapters into intermediate layers of the frozen SAM, where each adapter dynamically adjusts the effective rank of its weight matrix based on the input by selectively activating (rank-1) components of the matrix using a learned gating module. This adjustment enables fine-grained and input-aware robustification without compromising the generalization capability of SAM. Our model, GaRA-SAM, significantly outperforms prior work on all robust segmentation benchmarks. In particular, it surpasses the previous best IoU score by up to 21.3\\%p on ACDC, a challenging real corrupted image dataset.",
  "created": "2025-06-03",
  "updated": "2025-06-03",
  "id": "2506.02882",
  "categories": "cs.CV"
}

vs = AbstractVectorStore()
vs.initialize_store()
vs.add_single_abstract(
    abstract=metadata["abstract"],
    id=metadata["id"]
)