# CaterpillarTechathon_Alternative_Approach

The model is trained on bathroom section of NYUv2 dataset.

The model uses a physics-based approach for monocular depth estimation which refines predictions using the thin lens equation, d=f⋅B/D  By converting depth to disparity, the model applies physics constraints to correct depth inconsistencies. This ensures more realistic and physically accurate depth estimation.

Custom loss functions further enforce physical constraints. A depth loss maintains scale consistency, an edge-aware smoothness loss preserves object boundaries, and a physics consistency loss ensures planar depth in ground regions. These enhancements improve generalization, reduce errors, and handle sparse data better, making the approach ideal for autonomous driving and robotics.
