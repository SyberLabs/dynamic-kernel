"""
Domain Adapters — Pluggable Topology Definitions
==================================================
Each DomainAdapter encapsulates everything needed to instantiate a kernel
for a specific real-world environment: node features, edge structure,
semantic labels for features and intents, and display configuration.

Adding a new domain requires only adding a DomainAdapter instance to
ADAPTERS below — no changes to kernel.py or api.py needed.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from kernel import DynamicTopologyKernel, Topology, topology_from_edges


# ---------------------------------------------------------------------------
# Adapter schema
# ---------------------------------------------------------------------------

@dataclass
class DomainAdapter:
    """
    A self-describing topology definition.

    Attributes
    ----------
    key : str
        Machine-readable identifier used in API calls.
    name : str
        Human-readable display name.
    description : str
        One-sentence description of the environment.
    icon : str
        Emoji for UI display.
    accent : str
        Hex color for UI theming.
    undirected : bool
        If True, all edges are bidirectional. If False, directed.
    nodes : dict[str, list[float]]
        Maps node label -> feature vector [f0, f1, f2].
    edges : list[tuple[str, str, float]]
        (source, target, base_distance) triples.
    feature_labels : list[str]
        Human-readable names for each feature dimension.
        e.g. ["Fashion", "Food", "Tech"] or ["Speed", "Capacity", "Cost"]
    intent_presets : dict[str, list[float]]
        Named telemetry presets meaningful for this domain.
        e.g. {"Fashion Shopper": [1,0,0], "Foodie": [0,1,0], ...}
    node_bias_idx : int
        Index of the entry node that gets a baseline attractiveness boost.
    default_beta : float
        Default morphing strength applied uniformly at init.
    time_multiplier : float
        Scales distance -> transit ticks in PopulationSimulator.
    """
    key: str
    name: str
    description: str
    icon: str
    accent: str
    undirected: bool
    nodes: dict
    edges: List[Tuple]
    feature_labels: List[str]
    intent_presets: dict
    node_bias_idx: int = 0
    default_beta: float = 5.0
    time_multiplier: float = 4.0
    node_biases: Optional[dict] = None

    @property
    def N(self) -> int:
        return len(self.nodes)

    @property
    def F(self) -> int:
        return len(next(iter(self.nodes.values())))

    def build_topology(self) -> Topology:
        """Construct the Topology dataclass from this adapter's definition."""
        nodes_np = {k: np.array(v, dtype=np.float64) for k, v in self.nodes.items()}
        edges_typed = [(a, b, float(d)) for a, b, d in self.edges]
        return topology_from_edges(nodes=nodes_np, edges=edges_typed,
                                   undirected=self.undirected)

    def build_kernel(self, **kwargs) -> DynamicTopologyKernel:
        """
        Instantiate a fully configured DynamicTopologyKernel from this adapter.

        Any kwargs are forwarded to the kernel constructor, allowing callers
        to override temperature, feedback_rate, sponsor_decay, etc.
        """
        topo = self.build_topology()
        N = topo.N
        node_bias = np.zeros(N)
        if self.node_biases:
            label_to_idx = {label: i for i, label in enumerate(topo.labels)}
            for label, bias in self.node_biases.items():
                if label in label_to_idx:
                    node_bias[label_to_idx[label]] = float(bias)
        else:
            node_bias[self.node_bias_idx] = 0.3

        defaults = dict(
            topology=topo,
            alpha=1.0,
            beta=np.full((N, N), self.default_beta),
            feedback_rate=0.15,
            temperature=1.0,
            feedback_noise=0.02,
            node_bias=node_bias,
        )
        defaults.update(kwargs)
        return DynamicTopologyKernel(**defaults)

    def to_api_meta(self) -> dict:
        """Serializable metadata for the /api/topology/presets endpoint."""
        return {
            "key":           self.key,
            "name":          self.name,
            "description":   self.description,
            "icon":          self.icon,
            "accent":        self.accent,
            "undirected":    self.undirected,
            "nodeCount":     self.N,
            "featureLabels": self.feature_labels,
            "intentPresets": self.intent_presets,
        }


# ---------------------------------------------------------------------------
# Reference adapters
# ---------------------------------------------------------------------------

MALL = DomainAdapter(
    key="mall",
    name="Enterprise Mall",
    description="5-node shopping mall with fashion, food, and tech zones.",
    icon="🏬",
    accent="#4f8ef5",
    undirected=True,
    nodes={
        "Entrance":         [0.1, 0.1, 0.1],
        "Food Court":       [0.1, 0.9, 0.0],
        "Tech Store":       [0.0, 0.1, 0.9],
        "Premium Outlet A": [0.8, 0.0, 0.1],
        "Premium Outlet B": [0.9, 0.1, 0.0],
    },
    edges=[
        ("Entrance",         "Food Court",       5.0),
        ("Entrance",         "Tech Store",        4.0),
        ("Entrance",         "Premium Outlet A", 10.0),
        ("Food Court",       "Premium Outlet B",  6.0),
        ("Premium Outlet A", "Premium Outlet B",  3.0),
        ("Tech Store",       "Premium Outlet A",  7.0),
    ],
    feature_labels=["Fashion", "Food & Bev", "Technology"],
    intent_presets={
        "Fashion Shopper": [1.0, 0.0, 0.0],
        "Foodie":          [0.0, 1.0, 0.0],
        "Tech Enthusiast": [0.0, 0.0, 1.0],
        "Mixed Intent":    [0.33, 0.33, 0.33],
    },
)

AIRPORT = DomainAdapter(
    key="airport",
    name="Airport Terminal",
    description="Directed passenger flow from Security through gates and amenities.",
    icon="✈️",
    accent="#34d399",
    undirected=False,
    nodes={
        "Security":   [0.1, 0.1, 0.8],
        "Gate A":     [0.8, 0.1, 0.1],
        "Gate B":     [0.7, 0.2, 0.1],
        "Duty Free":  [0.9, 0.0, 0.1],
        "Restaurant": [0.1, 0.9, 0.0],
    },
    edges=[
        ("Security",    "Gate A",      8.0),
        ("Security",    "Gate B",      9.0),
        ("Security",    "Duty Free",   5.0),
        ("Security",    "Restaurant",  4.0),
        ("Restaurant",  "Gate A",      6.0),
        ("Restaurant",  "Gate B",      7.0),
        ("Duty Free",   "Gate A",      4.0),
        ("Duty Free",   "Gate B",      5.0),
    ],
    feature_labels=["Retail Spend", "F&B Spend", "Time Pressure"],
    intent_presets={
        "Business Traveller": [0.2, 0.1, 0.7],
        "Leisure Shopper":    [0.8, 0.1, 0.1],
        "Foodie Transit":     [0.1, 0.8, 0.1],
        "Neutral":            [0.33, 0.33, 0.33],
    },
)

MUSEUM = DomainAdapter(
    key="museum",
    name="Natural History Museum",
    description="Undirected visitor flow between exhibit halls, gift shop, and café.",
    icon="🏛️",
    accent="#9b6cf7",
    undirected=True,
    nodes={
        "Lobby":         [0.1, 0.1, 0.1],
        "Modern Art":    [0.9, 0.0, 0.1],
        "Ancient World": [0.1, 0.1, 0.8],
        "Gift Shop":     [0.7, 0.2, 0.1],
        "Café":          [0.1, 0.9, 0.0],
    },
    edges=[
        ("Lobby",        "Modern Art",    4.0),
        ("Lobby",        "Ancient World", 5.0),
        ("Lobby",        "Gift Shop",     3.0),
        ("Lobby",        "Café",          3.5),
        ("Modern Art",   "Gift Shop",     4.0),
        ("Ancient World","Gift Shop",     4.0),
        ("Café",         "Modern Art",    6.0),
    ],
    feature_labels=["Art Affinity", "Food & Rest", "History Interest"],
    intent_presets={
        "Art Lover":      [0.9, 0.0, 0.1],
        "History Buff":   [0.1, 0.1, 0.8],
        "Casual Visitor": [0.3, 0.4, 0.3],
        "School Group":   [0.2, 0.2, 0.6],
    },
)

SUPPLY_CHAIN = DomainAdapter(
    key="supply_chain",
    name="Supply Chain",
    description="Directed pipeline: Warehouse → Processing → Distribution → Retail endpoints.",
    icon="⛓️",
    accent="#f5a623",
    undirected=False,
    nodes={
        "Warehouse":    [0.0, 0.0, 1.0],
        "Processing":   [0.1, 0.1, 0.8],
        "Distribution": [0.4, 0.1, 0.5],
        "Retail A":     [0.9, 0.1, 0.0],
        "Retail B":     [0.8, 0.2, 0.0],
    },
    edges=[
        ("Warehouse",    "Processing",    3.0),
        ("Processing",   "Distribution",  4.0),
        ("Distribution", "Retail A",      5.0),
        ("Distribution", "Retail B",      6.0),
        ("Retail A",     "Warehouse",    12.0),  # return/restocking loop
        ("Retail B",     "Warehouse",    13.0),
    ],
    feature_labels=["Retail Demand", "Perishability", "Logistics Cost"],
    intent_presets={
        "High Demand":      [0.9, 0.1, 0.0],
        "Perishable Goods": [0.0, 0.9, 0.1],
        "Cost Optimised":   [0.1, 0.1, 0.8],
        "Balanced":         [0.33, 0.33, 0.33],
    },
    time_multiplier=3.0,
)

WHEEL_CITY = DomainAdapter(
    key="wheel_city",
    name="WHEEL City",
    description="12-node radial district city with a high-attraction CBD hub.",
    icon="wheel",
    accent="#38bdf8",
    undirected=True,
    nodes={
        "CBD": [0.10, 0.95, 0.70, 1.00],
        "North Residential": [0.85, 0.10, 0.20, 0.40],
        "South Residential": [0.80, 0.15, 0.15, 0.45],
        "East Residential": [0.70, 0.20, 0.25, 0.45],
        "West Residential": [0.75, 0.15, 0.20, 0.35],
        "Cultural Quarter": [0.25, 0.35, 0.90, 0.55],
        "Industrial Zone": [0.10, 0.55, 0.05, 0.80],
        "University": [0.45, 0.30, 0.75, 0.60],
        "Medical District": [0.35, 0.65, 0.20, 0.75],
        "Market District": [0.25, 0.85, 0.55, 0.70],
        "Transit Yard": [0.05, 0.45, 0.05, 0.95],
        "Waterfront": [0.30, 0.55, 0.80, 0.60],
    },
    edges=[
        ("CBD", "North Residential", 2.6),
        ("CBD", "South Residential", 2.8),
        ("CBD", "East Residential", 3.0),
        ("CBD", "West Residential", 3.2),
        ("CBD", "Cultural Quarter", 2.4),
        ("CBD", "Industrial Zone", 3.8),
        ("CBD", "University", 3.4),
        ("CBD", "Medical District", 2.2),
        ("CBD", "Market District", 2.0),
        ("CBD", "Transit Yard", 2.5),
        ("CBD", "Waterfront", 3.6),
        ("North Residential", "Cultural Quarter", 9.0),
        ("Cultural Quarter", "East Residential", 8.5),
        ("East Residential", "Waterfront", 10.0),
        ("Waterfront", "South Residential", 11.0),
        ("South Residential", "Industrial Zone", 8.0),
        ("Industrial Zone", "Transit Yard", 9.0),
        ("Transit Yard", "West Residential", 10.5),
        ("West Residential", "North Residential", 12.0),
        ("University", "Medical District", 8.0),
        ("Medical District", "Market District", 8.5),
    ],
    feature_labels=["Residential", "Commercial", "Cultural", "Transit"],
    intent_presets={
        "Resident": [0.7, 0.1, 0.2, 0.3],
        "Commuter": [0.1, 0.8, 0.0, 0.8],
        "Tourist": [0.0, 0.3, 0.9, 0.3],
        "Logistics": [0.0, 0.3, 0.0, 0.9],
        "Nightlife": [0.0, 0.4, 0.9, 0.2],
    },
    node_biases={"CBD": 0.6},
    time_multiplier=3.0,
)

RHIZOME_CITY = DomainAdapter(
    key="rhizome_city",
    name="RHIZOME City",
    description="12-node distributed small-world district mesh with redundant paths.",
    icon="rhizome",
    accent="#34d399",
    undirected=True,
    nodes={
        "Arts": [0.25, 0.35, 0.95, 0.40],
        "Market": [0.25, 0.90, 0.55, 0.65],
        "Port": [0.10, 0.55, 0.15, 0.90],
        "Park": [0.70, 0.10, 0.70, 0.35],
        "Square": [0.35, 0.70, 0.70, 0.70],
        "Station": [0.15, 0.65, 0.25, 0.95],
        "Old Town": [0.65, 0.25, 0.85, 0.45],
        "Library": [0.45, 0.20, 0.80, 0.55],
        "University": [0.55, 0.35, 0.75, 0.65],
        "Canal": [0.45, 0.35, 0.65, 0.55],
        "Clinic": [0.55, 0.50, 0.20, 0.70],
        "Workshops": [0.30, 0.60, 0.35, 0.75],
    },
    edges=[
        ("Arts", "Market", 3.2),
        ("Market", "Port", 4.1),
        ("Arts", "Park", 3.7),
        ("Market", "Square", 3.1),
        ("Port", "Station", 3.6),
        ("Park", "Square", 4.0),
        ("Square", "Station", 3.3),
        ("Park", "Old Town", 4.5),
        ("Square", "Library", 3.8),
        ("Station", "University", 4.2),
        ("Old Town", "Library", 3.4),
        ("Library", "University", 3.7),
        ("Arts", "Canal", 5.0),
        ("Canal", "Old Town", 3.8),
        ("Canal", "Market", 4.6),
        ("Library", "Clinic", 4.0),
        ("Clinic", "Station", 4.3),
        ("Clinic", "Workshops", 3.5),
        ("Workshops", "Port", 4.4),
        ("Workshops", "University", 4.1),
    ],
    feature_labels=["Residential", "Commercial", "Cultural", "Transit"],
    intent_presets={
        "Resident": [0.7, 0.1, 0.2, 0.3],
        "Commuter": [0.1, 0.8, 0.0, 0.8],
        "Tourist": [0.0, 0.3, 0.9, 0.3],
        "Logistics": [0.0, 0.3, 0.0, 0.9],
        "Nightlife": [0.0, 0.4, 0.9, 0.2],
    },
    node_biases={},
    time_multiplier=3.0,
)

NEURAL_DENSE = DomainAdapter(
    key="neural_dense",
    name="Neural Dense",
    description="12-node abstract dense neural topology for self-organizing beta dynamics.",
    icon="neural",
    accent="#9b6cf7",
    undirected=False,
    nodes={f"Neuron {i + 1}": [0.25, 0.25, 0.25, 0.25] for i in range(12)},
    edges=[
        (f"Neuron {i + 1}", f"Neuron {j + 1}", 1.0)
        for i in range(12) for j in range(12) if i != j
    ],
    feature_labels=["Type A", "Type B", "Type C", "Type D"],
    intent_presets={"Uniform": [0.25, 0.25, 0.25, 0.25]},
    default_beta=3.0,
    node_biases={},
    time_multiplier=1.0,
)

SOCIAL_MEDIA = DomainAdapter(
    key="social_media",
    name="Social Media Feed",
    description="16-cluster recommendation graph for attention routing, filter bubbles, and sponsored boosts.",
    icon="social",
    accent="#ef4444",
    undirected=False,
    nodes={
        # News, Entertainment, Practicality, Identity, Conflict, Novelty,
        # Social Proof, Credibility, Commercial, Depth.
        "Onboarding":          [0.30, 0.45, 0.35, 0.20, 0.05, 0.80, 0.45, 0.55, 0.20, 0.20],
        "Friend Updates":      [0.20, 0.55, 0.15, 0.65, 0.10, 0.35, 0.95, 0.60, 0.10, 0.25],
        "Local News":          [0.95, 0.20, 0.25, 0.35, 0.20, 0.35, 0.45, 0.75, 0.10, 0.45],
        "Science Explainers":  [0.55, 0.30, 0.75, 0.10, 0.05, 0.55, 0.25, 0.95, 0.05, 0.80],
        "Practical How-To":    [0.15, 0.30, 0.95, 0.10, 0.05, 0.45, 0.25, 0.80, 0.25, 0.60],
        "Sports Highlights":   [0.10, 0.90, 0.10, 0.35, 0.20, 0.55, 0.75, 0.55, 0.25, 0.20],
        "Comedy & Creators":   [0.05, 0.95, 0.10, 0.35, 0.15, 0.85, 0.80, 0.50, 0.35, 0.15],
        "Gaming Streams":      [0.05, 0.85, 0.20, 0.45, 0.25, 0.70, 0.85, 0.45, 0.35, 0.25],
        "Music & Culture":     [0.15, 0.85, 0.10, 0.50, 0.10, 0.65, 0.65, 0.60, 0.25, 0.35],
        "Wellness":            [0.10, 0.35, 0.70, 0.35, 0.05, 0.45, 0.35, 0.70, 0.40, 0.55],
        "Finance Advice":      [0.20, 0.25, 0.85, 0.20, 0.20, 0.40, 0.45, 0.65, 0.70, 0.60],
        "Civic Debate":        [0.85, 0.20, 0.35, 0.70, 0.55, 0.45, 0.55, 0.65, 0.05, 0.70],
        "Conflict Commentary": [0.65, 0.45, 0.10, 0.85, 0.95, 0.75, 0.85, 0.35, 0.15, 0.35],
        "Conspiracy/Rumor":    [0.55, 0.40, 0.05, 0.90, 0.90, 0.90, 0.75, 0.05, 0.25, 0.25],
        "Shopping Promoted":   [0.05, 0.45, 0.45, 0.20, 0.05, 0.55, 0.55, 0.45, 0.98, 0.10],
        "Longform Off-Ramp":   [0.70, 0.15, 0.55, 0.20, 0.05, 0.25, 0.15, 0.95, 0.05, 0.98],
    },
    edges=[
        ("Onboarding", "Friend Updates", 1.2),
        ("Onboarding", "Local News", 1.4),
        ("Onboarding", "Comedy & Creators", 1.0),
        ("Onboarding", "Practical How-To", 1.3),
        ("Onboarding", "Shopping Promoted", 1.7),
        ("Friend Updates", "Comedy & Creators", 1.1),
        ("Friend Updates", "Music & Culture", 1.2),
        ("Friend Updates", "Local News", 1.4),
        ("Friend Updates", "Civic Debate", 1.8),
        ("Local News", "Civic Debate", 1.0),
        ("Local News", "Science Explainers", 1.4),
        ("Local News", "Conflict Commentary", 1.6),
        ("Local News", "Longform Off-Ramp", 1.8),
        ("Science Explainers", "Practical How-To", 1.0),
        ("Science Explainers", "Longform Off-Ramp", 1.1),
        ("Science Explainers", "Local News", 1.6),
        ("Practical How-To", "Science Explainers", 1.1),
        ("Practical How-To", "Wellness", 1.2),
        ("Practical How-To", "Finance Advice", 1.4),
        ("Practical How-To", "Shopping Promoted", 1.7),
        ("Sports Highlights", "Comedy & Creators", 1.0),
        ("Sports Highlights", "Gaming Streams", 1.3),
        ("Sports Highlights", "Friend Updates", 1.4),
        ("Comedy & Creators", "Sports Highlights", 1.1),
        ("Comedy & Creators", "Gaming Streams", 1.2),
        ("Comedy & Creators", "Music & Culture", 1.0),
        ("Comedy & Creators", "Shopping Promoted", 1.6),
        ("Gaming Streams", "Comedy & Creators", 1.1),
        ("Gaming Streams", "Sports Highlights", 1.3),
        ("Gaming Streams", "Music & Culture", 1.4),
        ("Music & Culture", "Friend Updates", 1.3),
        ("Music & Culture", "Comedy & Creators", 1.1),
        ("Music & Culture", "Longform Off-Ramp", 2.0),
        ("Wellness", "Practical How-To", 1.1),
        ("Wellness", "Finance Advice", 1.6),
        ("Wellness", "Science Explainers", 1.7),
        ("Finance Advice", "Practical How-To", 1.3),
        ("Finance Advice", "Shopping Promoted", 1.2),
        ("Finance Advice", "Conflict Commentary", 2.0),
        ("Civic Debate", "Local News", 1.2),
        ("Civic Debate", "Conflict Commentary", 0.8),
        ("Civic Debate", "Science Explainers", 1.7),
        ("Civic Debate", "Longform Off-Ramp", 1.8),
        ("Conflict Commentary", "Civic Debate", 0.9),
        ("Conflict Commentary", "Conspiracy/Rumor", 1.0),
        ("Conflict Commentary", "Local News", 1.9),
        ("Conflict Commentary", "Longform Off-Ramp", 2.2),
        ("Conspiracy/Rumor", "Conflict Commentary", 0.8),
        ("Conspiracy/Rumor", "Civic Debate", 1.8),
        ("Conspiracy/Rumor", "Science Explainers", 2.3),
        ("Conspiracy/Rumor", "Longform Off-Ramp", 2.5),
        ("Shopping Promoted", "Practical How-To", 1.5),
        ("Shopping Promoted", "Comedy & Creators", 1.4),
        ("Shopping Promoted", "Friend Updates", 1.8),
        ("Longform Off-Ramp", "Science Explainers", 1.0),
        ("Longform Off-Ramp", "Local News", 1.3),
        ("Longform Off-Ramp", "Friend Updates", 1.8),
    ],
    feature_labels=[
        "News", "Entertainment", "Practicality", "Identity", "Conflict",
        "Novelty", "Social Proof", "Credibility", "Commercial", "Depth",
    ],
    intent_presets={
        "Casual Scroller":     [0.15, 0.75, 0.20, 0.30, 0.10, 0.70, 0.70, 0.45, 0.20, 0.20],
        "News Seeker":         [0.95, 0.15, 0.35, 0.30, 0.15, 0.35, 0.35, 0.85, 0.05, 0.55],
        "Civic Learner":       [0.85, 0.10, 0.45, 0.45, 0.20, 0.35, 0.25, 0.95, 0.05, 0.85],
        "High-Arousal Scroll": [0.55, 0.60, 0.05, 0.90, 1.00, 0.85, 0.90, 0.20, 0.10, 0.20],
        "Practical Research":  [0.25, 0.15, 0.95, 0.10, 0.05, 0.35, 0.20, 0.85, 0.20, 0.75],
        "Shopping Intent":     [0.05, 0.45, 0.45, 0.15, 0.05, 0.55, 0.55, 0.45, 1.00, 0.10],
        "Deep Research":       [0.70, 0.05, 0.65, 0.15, 0.05, 0.20, 0.10, 1.00, 0.05, 1.00],
    },
    default_beta=1.4,
    node_biases={"Onboarding": 0.3, "Friend Updates": 0.2, "Longform Off-Ramp": 0.1},
    time_multiplier=1.0,
)

# ---------------------------------------------------------------------------
# Registry — single source of truth for all adapters
# ---------------------------------------------------------------------------

ADAPTERS: dict[str, DomainAdapter] = {
    a.key: a for a in [
        MALL, AIRPORT, MUSEUM, SUPPLY_CHAIN, WHEEL_CITY, RHIZOME_CITY,
        NEURAL_DENSE, SOCIAL_MEDIA
    ]
}


def get_adapter(key: str) -> DomainAdapter:
    if key not in ADAPTERS:
        raise KeyError(
            f"Unknown adapter '{key}'. "
            f"Available: {list(ADAPTERS.keys())}"
        )
    return ADAPTERS[key]
