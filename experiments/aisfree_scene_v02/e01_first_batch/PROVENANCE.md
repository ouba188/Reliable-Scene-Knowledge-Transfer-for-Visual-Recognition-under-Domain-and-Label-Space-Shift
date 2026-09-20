# Provenance
- User E00 completed report: source of new dedup, mask and repaired-product completion counts. No raw table rescan in this environment.
- Repository commit: `435fa4b38e77d494c45ff6d91a6f0ed08f41c6ef`.
- Connector-read `knowledge/phase0/splits/lopo_folds.json`, ports block: source of all 329 product strings and adapt/eval roles.
- Connector-read `port_acquisition_split.csv`: source blob SHA `833428022446beb3a30894b7f5e8613c76281c1b`; local verification command compares all products/roles.
- Connector-read `experiments/aisfree_scene_v02/feature_permissions.yaml`: source blob SHA `786b0f8d9323bddf552f70292ce65cb6a741abba`.
- H/J/B/D, epochs, candidate sampler, relation transforms, calibration recipe and pilot ordering are assistant-proposed first-run settings, NOT extracted experiment findings.
- Only unit/synthetic smoke tests ran here. TEST_REPORT.json records scope.
- PyTorch official reproducibility guidance: https://docs.pytorch.org/docs/stable/notes/randomness.html ; fixed seeds do not ensure cross-release/device identity.
