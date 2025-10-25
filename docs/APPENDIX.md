# Appendix: Architecture & Benchmark Plan for the Bee‑Swarm TSP
(Condensed — see earlier message for full text.)

- Candidate graph: Delaunay ∪ k‑NN (then add α‑near when ready).
- Agents: 2‑opt/3‑opt (baseline uses 2‑opt) + double‑bridge kicks; Lévy‑style restarts optional.
- EHM memory: edge histogram with smoothing; scouts sampled from it.
- Integrator: POPMUSIC‑style merge or EAX; baseline ships POPMUSIC‑style.
- Zones with 10–20% overlap via k‑means boundary margin heuristic.
