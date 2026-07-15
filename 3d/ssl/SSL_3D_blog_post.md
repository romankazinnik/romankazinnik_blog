# The Labeling Problem: Self-Supervision for 3-D Point Clouds

Hand-labeling a building scan is the most expensive artifact in 3-D machine learning — and every new site asks for it again. This case study applies the LLM recipe to a real LiDAR scan: pretrain on raw geometry and color first, then spend labels only on a small classification head. One building, 4M points, 7 structural classes, one GPU workstation, one honest baseline.

- **The baseline fails.** A classifier on raw xyz + rgb + hand-crafted geometry: **mIoU 0.508**. Coordinates memorize *this* building; color varies with surface finish, not structural role.
- **Phase 1 — pretraining, zero labels.** Mask each point's color and local shape statistics; make the network predict them from spatial context (BERT/MAE-style, not GPT — point clouds have no token order). 77 epochs → a 256-d embedding per point.
- **Phase 2 — few labels.** Freeze the encoder; train a small MLP head on 786K balanced points (20% of labels). Same head, same label budget as the baseline — only the representation changed.
- **Result: mIoU 0.508 → 0.866** (+70% relative). **96.7%** of 3.15M *held-out* points correct; train–eval gap under one point — the representation is doing the work, not memorization.
- **The failures have geometry.** Nearly all errors sit on **edges** — wall–slab junctions, member borders — where local descriptors average across the boundary. Diagnosis in hand: edge-sensitive latent features and boundary-aware sampling are next.
- **The label-efficiency curve is the central result.** Sweeping the label budget: SSL hits mIoU **0.694 with 1% of labels — beating raw features given 20× more** (0.508 at 20%). SSL saturates by 10% (0.856 of the 0.866 ceiling), while the raw curve barely moves: 200× more labels lift it only 0.297→0.508. More annotation cannot fix a weak representation.
- **Coming next:** cross-site transfer — pretraining on an unlabeled second building, classifying it with the first building's head.

Deck with the full visual story below. If you are building AI for the physical world — reach out.
