# Cross-port SAR ship classification with scene knowledge — dataset and protocol card

Everything in this directory is reproducible from the scripts listed at the bottom, on a Windows box with the
objects/scenes/OSM assets described under *Provenance*. Nothing here depends on the (now retired) training server.

## 1. What the dataset is

| item | value |
|---|---|
| chips | **30,496** at 224×224, uint8, VV and VH stored as paired memmaps (`dataset244/<port>_<POL>.npy`) |
| ports | **8** — Jebel Ali 11,052 · Antwerp-Bruges 8,423 · Hamburg 4,839 · Fujairah 4,162 · Houston 1,201 · Busan 539 · Callao 156 · Los Angeles 124 |
| classes | **17** — the eight the closed-set work uses (bulk_carrier, fishing_vessel, general_cargo, product_chemical_tanker, container_ship, crude_oil_tanker, tug_towing, offshore_supply) plus nine held-out types (lpg_lng_tanker 2,387 · dredger 1,923 · ro_ro_vehicle_carrier 1,291 · passenger_ship 1,240 · pleasure_craft 781 · pilot_port_tender 392 · heavy_load_carrier 370 · sailing_vessel 302 · reefer_cargo 46) |
| index | `dataset244/index.csv` — port, pol, class, level, row, product, det, px, py, ais, pre, fine, src, conf |

**Label tiers are recorded, not resolved.** `class` is the first non-empty of `fine` / `pre` / `ais`; the
`src`/`conf`/`level` columns let an analysis stratify. This matters: the closed-set pool used an AIS-matched tier,
whereas this build inherits a mixed tier, and the tier difference alone moves closed-set balanced accuracy from
~0.44 to ~0.21 (measured). **Do not mix the two tiers without saying so.**

## 2. Provenance and licensing

| asset | source | note |
|---|---|---|
| SAR scenes | Sentinel-1 GRD, geocoded 8-bit UTM (`F:/SAR_0922/<port>/{VV,VH}/<product>_<POL>_UTM_8bit.tif`) | Copernicus; attribution required |
| detection objects | `knowledge_set_841/objects/objects_classed.csv.gz` (2.90 M rows) | carries `object_id = <product>\|<POL>\|<index>\|<det>` and `world_x/world_y` |
| OSM facilities | local extracts under `E:/Install_packs/port_osm_run/pbf_2025/` | ODbL; region extracts, `hamburg-250101` being city-level |
| port harbour coordinates | public coordinates, hard-coded in `e92_facilities_all.py` | sanity-checked against the region facility hotspots (7.8 / 11.8 / 32.6 km) |

The crops come from the object table's `world_x/world_y` through each scene's geotransform — the one mapping
verified end to end (a crop centred on its ship, and a same-object VV/VH pair landing within 2 px).

## 3. Protocol

* **Leave-one-port-out** over the eight ports. A fold trains on the other seven and evaluates on the held-out port.
* **Metrics.** Balanced accuracy (macro recall over the classes present in that port) is primary; plain accuracy
  is always reported alongside, because the two disagree in sign on 3 of 24 ports of the wider pool (class
  composition, not noise). Per-class results are reported as **cross-port sign counts** (in how many ports a class
  moves up or down) rather than port averages — the effect's sign is a class property, and port averages hide it.
* **Unknown classes never train.** The nine held-out types are excluded from every fitting step and appear only at
  evaluation. Novelty supervision, when used, is built from SOURCE-side pseudo-unknowns (two of the eight known
  classes held out), never from the target.
* **Label-free target usage.** Transductive steps (within-port percentile transform of the knowledge, per-fold PCA
  and feature standardisation) fit on training sources or on unlabelled target features only.

## 4. Baselines (same folds, cached 2048-d ResNet50-S1 features)

| arm | known-class BA | note |
|---|---:|---|
| visual ridge | 0.209 | this dataset's raw build |
| visual + facility knowledge ridge | ~0.21 | knowledge carries the signal in feature space |
| KOSR (3 heads + coupling law) | 0.1973 | e90/e91; the coupling law is **unverified** (see §6) |
| OOD novelty (energy / Mahalanobis) | AUC 0.493 | chance — unknown types are not out-of-distribution |

## 5. What is solid

* **High-confidence absorption**: a closed-set recogniser is as confident on unseen ship types (mean 0.238) as on
  known ones (0.233), and **85.5 %** of unseen chips cross a threshold set to keep 90 % of correct known
  predictions. This is the open-set motivation, measured.
* **Feature-space knowledge effect**: the class-sharing score of the knowledge dims discriminates harm from rescue
  at AUC 0.72–0.75, with the signal confined to the *contrast* block (dims 14–41).
* **Port-local facility extraction**: 8 ports, 763–14,414 facility points each, validated by a metres-level
  nearest-facility distance after calibration (3 / 4 / 4 / 9 / 19 / 81 / 174 / 596 m).
* **Negative results with clean controls**: domain-level gating, source selection, label-shift correction,
  pairwise structure, added gate features, joint end-to-end training — each rejected with a paired test.

## 6. Known defects — read before using

1. **Port attribution is regional.** The object table's `port` field assigns objects over the whole acquisition
   footprint, so only ~1.5 % of chips sit within 3 km of a port facility and the object-density "centre" of a port
   is 38–140 km off. Any per-chip port-area analysis must filter by distance to facilities first.
2. **The label tier is mixed** (§1) and the raw build lacks the quality filters (screen status, confidence tier,
   dedup) that the older 8-class pool had; that is why the base accuracy here (0.21) is about half of the other
   pool's (0.44). The filters exist in the object table and have not been applied yet.
3. **The physical-adjacency form of the mechanism does not hold**: after density normalisation only a partial
   class ordering survives (crude 2/16, lpg 6/16, bulk 11/16, container 13/16, fishing 16/16). The discriminating
   quantity is the feature-level contrast, not physical proximity.
4. **The coupling law is unverified.** Both a trained sigmoid novelty head and training-free OOD scores fail to
   identify novel chips; §5's convergence argument (novelty must come from knowledge-side unexplainability) is an
   argument, not yet a result.
5. **8 ports of 24.** The other sixteen await the ongoing scene-copy onto `F:`; `e84_build_224.py` resumes and
   fills them automatically because its resume key is the index, and zero-row ports are retried.

## 7. Scripts

| script | role |
|---|---|
| `e83_utm_map.py` | proves the object→scene→crop mapping (the only verified one) |
| `e84_build_224.py` | builds the dataset (tolerant to mid-copy scenes, resumable, incremental index) |
| `e86/e92_facilities_all.py` | port-local facility extraction from local OSM pbf |
| `e93_calib.py` | per-port calibration of `world_x/world_y` onto lon/lat, accepted on the minimum distance |
| `e87/e94/e95_density_norm.py` | proximity and density-normalised proximity tables |
| `e88/e89/e96_ood_gate.py` | the knowledge-side NAR attempts, including the OOD failure |
| `kosr.py`, `e90/e91_kosr_*.py` | the KOSR module, its self-check, and the two runs |
