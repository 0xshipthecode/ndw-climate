#!/usr/bin/env bash
python scripts/publication_figures_triangles.py NINO3.4 SAT 1 SLP 2 false nino3.4_sat_1_slp_1.eps
python scripts/publication_figures_triangles.py SOI SAT 1 SLP 2 false soi_sat_1_slp_1.eps
python scripts/publication_figures_triangles.py NAO_station SAT 6 SLP 6 true nao_station_sat_6_slp_6.eps
python scripts/publication_figures_triangles.py NAO_pca SAT 6 SLP 6 true nao_pca_sat_6_slp_6.eps
python scripts/publication_figures_triangles.py PNA SLP 10 SLP 12 false pna_slp_10_slp_12.eps
python scripts/publication_figures_triangles.py EA SAT 24 SLP 21 true ea_sat_24_slp_21.eps
python scripts/publication_figures_triangles.py WP SAT 34 SLP 12 false wp_sat_34_slp_12.eps



