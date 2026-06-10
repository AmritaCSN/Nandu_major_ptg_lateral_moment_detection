"""
ptg_scorer.py
-------------

PTS, EPL, and IPI scoring for PTG paths.

Definitions:
- PTS(p, id) = sum_i -log P(lambda_i | id)
  Identity-specific path rarity

- EPL(p) = sum_i -log P(lambda_{i+1} | lambda_i)
  Global sequence abnormality (Markov transitions)

- IPI(id, T) = number of distinct hosts reached through lateral movement in T

Final score:
- ALERT = alpha * PTS + beta * EPL + gamma * IPI

Notes:
- Baseline probabilities are trained on benign events only.
- IPI follows identity -> session -> host paths, and also supports direct
  identity -> host edges where present.
"""

import math
from collections import defaultdict


LAMBDA = [
    "login",
    "remote_login",
    "lateral",
    "escalation",
    "token_reuse",
    "group_add",
]

LATERAL_EDGES = {"lateral", "remote_login"}


class Baseline:
    """
    Learn Laplace-smoothed:
    - P(edge_type | identity)
    - P(curr_edge | prev_edge)
    from benign events only.
    """

    def __init__(self, smoothing=1.0, unseen_floor=1e-4):
        self.s = smoothing
        self.unseen_floor = unseen_floor
        self.p_edge = {}
        self.markov = {}
        self._num_edge_types = len(LAMBDA)

    def train(self, df):
        required_columns = {"src_user", "time", "edge_type"}
        missing = sorted(required_columns - set(df.columns))
        if missing:
            raise ValueError(f"Baseline.train() missing required columns: {missing}")

        identity_counts = defaultdict(lambda: defaultdict(int))
        markov_counts = defaultdict(lambda: defaultdict(int))

        grouped = df.sort_values(["src_user", "time"]).groupby("src_user", sort=False)

        for identity, group in grouped:
            edge_types = group["edge_type"].tolist()

            for edge_type in edge_types:
                identity_counts[identity][edge_type] += 1

            for i in range(len(edge_types) - 1):
                prev_edge = edge_types[i]
                curr_edge = edge_types[i + 1]
                markov_counts[prev_edge][curr_edge] += 1

        for identity, counts in identity_counts.items():
            total = sum(counts.values()) + self.s * self._num_edge_types
            self.p_edge[identity] = {
                edge_type: (counts.get(edge_type, 0) + self.s) / total
                for edge_type in LAMBDA
            }

        for prev_edge, counts in markov_counts.items():
            total = sum(counts.values()) + self.s * self._num_edge_types
            self.markov[prev_edge] = {
                edge_type: (counts.get(edge_type, 0) + self.s) / total
                for edge_type in LAMBDA
            }

    def pe(self, identity, edge_type):
        """
        Return P(edge_type | identity).

        Known identities get their smoothed learned distribution.
        Unseen identities get a strong rarity floor.
        """
        identity_distribution = self.p_edge.get(identity)
        if identity_distribution is None:
            return self.unseen_floor
        return identity_distribution.get(edge_type, self.unseen_floor)

    def pt(self, prev_edge, curr_edge):
        """
        Return P(curr_edge | prev_edge) from the global Markov model.
        """
        transition_distribution = self.markov.get(prev_edge)
        if transition_distribution is None:
            return self.unseen_floor
        return transition_distribution.get(curr_edge, self.unseen_floor)


class PTGScorer:
    def __init__(self, baseline, alpha=0.4, beta=0.4, gamma=0.2):
        self.bl = baseline
        self.a = alpha
        self.b = beta
        self.g = gamma

    def pts(self, identity, edge_sequence):
        """
        Identity-specific path rarity.
        """
        if not edge_sequence:
            return 0.0
        return sum(-math.log(self.bl.pe(identity, edge_type)) for edge_type in edge_sequence)

    def epl(self, edge_sequence):
        """
        Global sequence abnormality using edge-to-edge Markov transitions.
        """
        if len(edge_sequence) < 2:
            return 0.0

        return sum(
            -math.log(self.bl.pt(edge_sequence[i], edge_sequence[i + 1]))
            for i in range(len(edge_sequence) - 1)
        )

    def ipi(self, ptg, identity):
        """
        Count distinct hosts reached via lateral movement from a given identity.

        PTG traversal supported:
        - id -> host
        - id -> session -> host

        A host is counted if the relevant edge path is classified as lateral.
        """
        graph = ptg.G
        identity_node = f"id:{identity}"

        if identity_node not in graph:
            return 0

        reached_hosts = set()

        for _, mid_node, edge_data_1 in graph.out_edges(identity_node, data=True):
            mid_node_type = graph.nodes[mid_node].get("node_type")

            if mid_node_type == "host":
                if edge_data_1.get("edge_type") in LATERAL_EDGES:
                    reached_hosts.add(graph.nodes[mid_node]["label"])

            elif mid_node_type == "session":
                for _, host_node, edge_data_2 in graph.out_edges(mid_node, data=True):
                    if graph.nodes[host_node].get("node_type") != "host":
                        continue

                    if (
                        edge_data_1.get("edge_type") in LATERAL_EDGES
                        or edge_data_2.get("edge_type") in LATERAL_EDGES
                    ):
                        reached_hosts.add(graph.nodes[host_node]["label"])

        return len(reached_hosts)

    def score_path(self, ptg, path):
        """
        Score one extracted PTG path.
        """
        identity = ptg.G.nodes[path["root"]]["label"]
        edge_sequence = path["edges"]

        pts_score = self.pts(identity, edge_sequence)
        epl_score = self.epl(edge_sequence)
        ipi_score = self.ipi(ptg, identity)
        alert_score = self.a * pts_score + self.b * epl_score + self.g * ipi_score

        return {
            "identity": identity,
            "edges": edge_sequence,
            "times": path["times"],
            "event_ids": path.get("event_ids", []),
            "PTS": round(pts_score, 4),
            "EPL": round(epl_score, 4),
            "IPI": ipi_score,
            "ALERT": round(alert_score, 4),
        }

    def score_snapshot(self, ptg, max_depth=4):
        """
        Score all extracted identity-rooted paths in one PTG snapshot.
        """
        scored_paths = []
        for path in ptg.extract_identity_paths(max_depth=max_depth):
            scored_paths.append(self.score_path(ptg, path))
        return scored_paths