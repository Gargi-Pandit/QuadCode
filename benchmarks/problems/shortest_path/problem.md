# Shortest path in a weighted graph

Write a Python function `shortest_path(n: int, edges: list[list[int]], src: int, dst: int) -> int`.

The graph is directed with nodes `0..n-1`. Each edge is `[u, v, w]`: an edge from `u` to `v` with weight `w >= 0`. Return the minimum total weight of a path from `src` to `dst`.

- Return 0 if `src == dst`, and -1 if `dst` cannot be reached.
- Parallel edges and self-loops may occur. Up to 10^5 nodes and 2 * 10^5 edges.
