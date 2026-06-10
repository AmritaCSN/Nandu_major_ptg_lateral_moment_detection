"""
ptg_graph.py
------------

Privilege Transition Graph (PTG) as a first-class structure.

G_PTG = (V, E, T, lambda, tau)

V = V_id ∪ V_host ∪ V_session
E ⊆ V × V × Λ
Λ = {login, remote_login, lateral, escalation, token_reuse, group_add}

This is a directed temporal multigraph.

Identity authenticates through a logon session to a host:
    id --lambda--> session --lambda--> host

When no session id is present, a direct:
    id --lambda--> host
edge is used.

LANL note:
- LANL auth provides login / lateral / remote_login from real columns.
- escalation / token_reuse / group_add are not fabricated for LANL.
- The graph supports the full edge vocabulary, but only the observed subset is populated.
"""

from collections import defaultdict

import networkx as nx


LAMBDA = [
    "login",
    "remote_login",
    "lateral",
    "escalation",
    "token_reuse",
    "group_add",
]

LATERAL_EDGES = {"lateral", "remote_login"}


def id_node(identity):
    return f"id:{identity}"


def host_node(host):
    return f"host:{host}"


def sess_node(session_id):
    return f"sess:{session_id}"


class PTG:
    """
    A single temporal snapshot graph for window [t_start, t_end).
    """

    def __init__(self, t_start=None, t_end=None):
        self.G = nx.MultiDiGraph()
        self.t_start = t_start
        self.t_end = t_end

    def add_event(
        self,
        ts,
        identity,
        src_comp,
        dst_comp,
        edge_type,
        session_id=None,
        event_id=None,
    ):
        identity_node = id_node(identity)
        if not self.G.has_node(identity_node):
            self.G.add_node(identity_node, node_type="identity", label=identity)

        destination_host_node = host_node(dst_comp)
        if not self.G.has_node(destination_host_node):
            self.G.add_node(destination_host_node, node_type="host", label=dst_comp)

        if session_id:
            session_node = sess_node(session_id)
            if not self.G.has_node(session_node):
                self.G.add_node(session_node, node_type="session", label=session_id)

            self.G.add_edge(
                identity_node,
                session_node,
                edge_type=edge_type,
                timestamp=ts,
                src_host=src_comp,
                event_id=event_id,
            )
            self.G.add_edge(
                session_node,
                destination_host_node,
                edge_type=edge_type,
                timestamp=ts,
                src_host=src_comp,
                event_id=event_id,
            )
        else:
            self.G.add_edge(
                identity_node,
                destination_host_node,
                edge_type=edge_type,
                timestamp=ts,
                src_host=src_comp,
                event_id=event_id,
            )

    def _make_path_record(self, root, node_path, edge_types, timestamps, event_ids):
        return {
            "nodes": node_path,
            "edges": edge_types,
            "times": timestamps,
            "event_ids": event_ids,
            "root": root,
        }

    def extract_identity_paths(self, max_depth=4):
        """
        For each identity node, walk forward following edges in non-decreasing
        timestamp order, up to max_depth edges.

        Returns a list of dictionaries containing:
        - nodes
        - edges
        - times
        - event_ids
        - root

        This is a bounded DFS and replaces all_simple_paths over all node pairs.
        """
        paths = []

        identity_nodes = [
            node
            for node, node_data in self.G.nodes(data=True)
            if node_data.get("node_type") == "identity"
        ]

        for root in identity_nodes:
            stack = [(root, [root], [], [], [], -float("inf"))]

            while stack:
                current_node, node_path, edge_types, timestamps, event_ids, last_ts = stack.pop()

                if len(edge_types) >= max_depth:
                    if len(edge_types) >= 1:
                        paths.append(
                            self._make_path_record(root, node_path, edge_types, timestamps, event_ids)
                        )
                    continue

                outgoing_edges = list(self.G.out_edges(current_node, data=True))

                if not outgoing_edges:
                    if len(edge_types) >= 1:
                        paths.append(
                            self._make_path_record(root, node_path, edge_types, timestamps, event_ids)
                        )
                    continue

                extended = False

                for _, next_node, edge_data in outgoing_edges:
                    ts = edge_data.get("timestamp", 0)

                    if ts >= last_ts and next_node not in node_path:
                        extended = True
                        stack.append(
                            (
                                next_node,
                                node_path + [next_node],
                                edge_types + [edge_data.get("edge_type")],
                                timestamps + [ts],
                                event_ids + [edge_data.get("event_id")],
                                ts,
                            )
                        )

                if not extended and len(edge_types) >= 1:
                    paths.append(
                        self._make_path_record(root, node_path, edge_types, timestamps, event_ids)
                    )

        return paths

    def stats(self):
        node_type_counts = defaultdict(int)
        for _, node_data in self.G.nodes(data=True):
            node_type_counts[node_data.get("node_type")] += 1

        edge_type_counts = defaultdict(int)
        for _, _, edge_data in self.G.edges(data=True):
            edge_type_counts[edge_data.get("edge_type")] += 1

        return {
            "nodes": dict(node_type_counts),
            "edges": dict(edge_type_counts),
            "n_nodes": self.G.number_of_nodes(),
            "n_edges": self.G.number_of_edges(),
        }


def build_snapshots(df, delta_seconds=3600, stride_seconds=None, session_col=None):
    """
    Slide a time window across the labeled events and build one PTG per window.

    Required dataframe columns:
    - time
    - src_user
    - src_comp
    - dst_comp
    - edge_type

    Optional:
    - session column passed via session_col
    """
    if df.empty:
        return []

    required_columns = {"time", "src_user", "src_comp", "dst_comp", "edge_type"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"build_snapshots() missing required columns: {missing}")

    if session_col is not None and session_col not in df.columns:
        raise ValueError(f"Requested session column '{session_col}' not found in dataframe")

    stride = stride_seconds or (delta_seconds / 2)

    time_min = int(df["time"].min())
    time_max = int(df["time"].max())

    df = df.sort_values("time")
    times = df["time"].to_numpy()

    snapshots = []
    current_start = time_min

    while current_start <= time_max:
        current_end = current_start + delta_seconds
        mask = (times >= current_start) & (times < current_end)
        window_df = df[mask]

        if len(window_df) > 0:
            ptg = PTG(current_start, current_end)

            for row in window_df.itertuples(index=True):
                session_id = getattr(row, session_col) if session_col else None
                ptg.add_event(
                    row.time,
                    row.src_user,
                    row.src_comp,
                    row.dst_comp,
                    row.edge_type,
                    session_id=session_id,
                    event_id=row.Index,
                )

            if ptg.G.number_of_edges() > 0:
                snapshots.append(ptg)

        current_start += stride

    return snapshots