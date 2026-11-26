# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Zenodo metadata for the research-data release (T6.3)

**Status: uploaded and published by the author on 2026-11-26.** DOI 10.5281/zenodo.24551101
(https://doi.org/10.5281/zenodo.24551101), version 1.0.0, resource type Dataset, licence CC BY 4.0 (code MIT inside
the code zip), concept DOI 10.5281/zenodo.24551100. The 8 published files match the local `release/zenodo/` files
by MD5 and size (checked through the Zenodo API on 2026-11-26): `xai-dose-response_code_0e128c4.zip`,
`xai-dose-response_results_0e128c4.zip`, `xai-dose-response_checkpoints_part1of4.zip` to `part4of4.zip`,
`MANIFEST.md` and `README.md`. The DOI is now in the paper, `CITATION.cff`, `RELEASE_README.md`, `declarations.md`
and the cover letter. The archived `README.md` keeps the DOI placeholder because the DOI did not exist at upload time.

The fields and steps below are kept as the record of what was entered.

| Field | Value |
|---|---|
| Resource type | Software (or Dataset; Zenodo allows one type. Choose **Dataset** if the DOI is cited as research data, because the journal asks for data deposition; the code is in the same record.) |
| Title | Data and code for: A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers |
| Creators | Shaikh, Md Junaid Faizan; affiliation Khaja Bandanawaz University; ORCID 0009-0002-6565-1835 |
| Publication date | date of upload |
| Version | 1.0.0 |
| Licence | Code: MIT. Data, results and checkpoints: CC BY 4.0. Zenodo takes one licence per record: select **CC BY 4.0** in the licence field and state in the description that the code is under the MIT licence (`LICENSE` in the code zip). |
| Keywords | Explainable AI; Faithfulness metrics; Perturbation metrics; Shortcut learning; Meta-evaluation; Pre-registration; Deep learning; Image classification |
| Language | English |
| Related identifiers | 10.5281/zenodo.24285532 (is supplemented by); 10.5281/zenodo.24333540 (is supplemented by); 10.5281/zenodo.24416017 (is supplemented by); 10.5281/zenodo.24495658 (is supplemented by); 10.5281/zenodo.24285531, the concept DOI of the pre-registration record (references); https://github.com/JunaidFaizan/xai-dose-response (is supplement to, URL); the journal article DOI after acceptance (is supplement to) |
| Files | `xai-dose-response_code_0e128c4.zip`, `xai-dose-response_results_0e128c4.zip`, `xai-dose-response_checkpoints_part1ofN.zip` ... `partNofN.zip`, `MANIFEST.md`, `README.md` |
| Funding | none (self-funded; the paper states no specific grant) |

Description (paste):

> Code, per-model results, analysis tables and trained checkpoints for the study "A Dose-Response Evaluation of
> Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers". 293 CNN image classifiers
> (Oxford-IIIT Pet, CIFAR-10, MNIST, PneumoniaMNIST) were trained with a signed training dose that sets how strongly
> each model relies on an injected shortcut; deletion, insertion and exact ROAD metrics were scored on fixed oracle
> explanation maps. The pre-registered predictions P1 to P13 and amendments A1 to A3 are in the related Zenodo
> records. Contents: the code (MIT licence), the results of `results/dose_response` (per-model CSV files, analysis
> tables, figures) and 297 checkpoints with their metrics files (CC BY 4.0). The raw images of MNIST,
> PneumoniaMNIST, CIFAR-10 and Oxford-IIIT Pet are not redistributed; download them from their owners (see README.md).
> `xai/road.py` contains a port of the MIT-licensed code of https://github.com/tleemann/road_evaluation.

Upload steps for the author:

1. The package was built with `.venv\Scripts\python.exe paper\make_release_package.py 0e128c4` (code zip from commit `0e128c4`; it rewrites `release/zenodo/`). Rebuild only if you change the code again.
2. On zenodo.org choose New upload, add all files from `release/zenodo/` (about 6 files plus the checkpoint parts,
   far below the 100 file and 50 GB limits).
3. Paste the fields above, check the SHA-256 values of the files against `MANIFEST.md`, publish.
4. Send the DOI to the agent. It replaces the DOI placeholder and the author runs `build_paper.py --final`. (Done:
   DOI 10.5281/zenodo.24551101, 2026-11-26.)
