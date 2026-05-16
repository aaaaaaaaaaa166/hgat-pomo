# Recommendation For 80/50

## Decision

- Recommend `combined_D / oracle_best_on_time` as the smallest stable observed feasible configuration.
- Business settings: response_window=5.0, delivery_window_extension=+3.0, resources=2.
- Continue model training: no.
- Continue teacher or ServicePolicy work: no.
- Recommended lever: change business rules and resource configuration.

## Boundary Explanation

- Response window alone is not enough: the best response-only high-acceptance row was `response_window_3.0 / oracle_best_acceptance` with acc=0.867 and on_time=0.291.
- Delivery window alone is not enough: the best delivery-only on-time row was `delivery_window_plus_8.0 / oracle_best_acceptance` with acc=0.600 and on_time=0.557.
- Resources alone are not enough: the best resource-only on-time row was `resource_count_4 / oracle_best_acceptance` with acc=0.588 and on_time=0.550.
- Lower combined settings A/B/C did not clear both targets; the closest lower setting was `combined_C / oracle_best_acceptance` with acc=0.849 and on_time=0.482.
- The target is primarily a business-configuration target, not a model-training target under the original constraints.
- If resources must be reduced below the recommended pair, current evidence only shows that response=5.0 and due+3.0 with one resource is insufficient; a separate focused sweep is needed to quantify the extra window.
- If time windows cannot be relaxed, current evidence only shows that up to five resources in the single-axis run does not reach 80/50; additional resources or acceptance-rule changes would need a separate feasibility run.

## Key Oracle Rows

- combined_D / oracle_best_acceptance eval=50: acc=0.954667, on_time=0.480447, late=744, avg_late=21.409174, max_late=63.515615, energy=374.654798, distance=6211.427027, hard=0
- combined_D / oracle_best_on_time eval=50: acc=0.914667, on_time=0.518950, late=660, avg_late=24.145694, max_late=66.110448, energy=396.888182, distance=6401.769881, hard=0
- combined_E / oracle_best_acceptance eval=50: acc=0.912667, on_time=0.588751, late=563, avg_late=14.922360, max_late=48.792923, energy=394.365402, distance=6931.750888, hard=0
- combined_E / oracle_best_on_time eval=50: acc=0.887333, on_time=0.633358, late=488, avg_late=16.029882, max_late=53.213413, energy=412.534480, distance=7041.879661, hard=0
- combined_F / oracle_best_acceptance eval=50: acc=0.994667, on_time=0.536863, late=691, avg_late=21.391745, max_late=71.106346, energy=395.983711, distance=6504.047871, hard=0
- combined_F / oracle_best_on_time eval=50: acc=0.989333, on_time=0.545148, late=675, avg_late=24.699792, max_late=71.806114, energy=420.281275, distance=6759.799542, hard=0
- combined_G / oracle_best_acceptance eval=50: acc=0.978667, on_time=0.608311, late=575, avg_late=15.854506, max_late=52.413750, energy=431.273171, distance=7517.838690, hard=0
- combined_G / oracle_best_on_time eval=50: acc=0.968667, on_time=0.629043, late=539, avg_late=17.683158, max_late=52.522568, energy=456.362742, distance=7778.122110, hard=0
- combined_D / oracle_best_acceptance eval=100: acc=0.936667, on_time=0.487189, late=1441, avg_late=21.240214, max_late=67.483751, energy=734.287057, distance=12187.253601, hard=0
- combined_D / oracle_best_on_time eval=100: acc=0.911667, on_time=0.524314, late=1301, avg_late=23.620672, max_late=66.221815, energy=781.492434, distance=12590.779830, hard=0
- combined_E / oracle_best_acceptance eval=100: acc=0.909667, on_time=0.595456, late=1104, avg_late=14.975779, max_late=50.832591, energy=779.694026, distance=13659.819407, hard=0
- combined_E / oracle_best_on_time eval=100: acc=0.874333, on_time=0.641250, late=941, avg_late=16.163347, max_late=53.213413, energy=815.417643, distance=13943.987848, hard=0
- combined_F / oracle_best_acceptance eval=100: acc=0.991000, on_time=0.529095, late=1400, avg_late=21.549208, max_late=71.931582, energy=783.316289, distance=12927.607900, hard=0
- combined_F / oracle_best_on_time eval=100: acc=0.985667, on_time=0.543118, late=1351, avg_late=24.343071, max_late=71.806114, energy=847.598227, distance=13575.179402, hard=0
- combined_G / oracle_best_acceptance eval=100: acc=0.971000, on_time=0.608994, late=1139, avg_late=15.588314, max_late=54.819866, energy=863.918517, distance=14943.782503, hard=0
- combined_G / oracle_best_on_time eval=100: acc=0.966000, on_time=0.632160, late=1066, avg_late=17.490370, max_late=52.522568, energy=905.748768, distance=15389.938741, hard=0
