# A Dose-Response Evaluation of Explanation Metrics under Controlled Shortcut Reliance in CNN Image Classifiers

## Zenodo metadata draft for the research-data release (T6.3)

Nothing is uploaded by the agent. Create a new Zenodo record (not a new version of the pre-registration
record), paste the fields below, upload the files named in `release/zenodo/MANIFEST.md`, publish, then send the
DOI so that `[[ZENODO_RELEASE_DOI]]` can be replaced in the paper, `CITATION.cff`, `RELEASE_README.md` and the
cover letter.

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
| Files | `xai-dose-response_code_<sha>.zip`, `xai-dose-response_results_<sha>.zip`, `xai-dose-response_checkpoints_part1ofN.zip` ... `partNofN.zip`, `MANIFEST.md`, `README.md` |
| Funding | none stated; fill only if the author declares funding |

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

1. Run `.venv\Scripts\python.exe paper\make_release_package.py` after the last commit (it rewrites `release/zenodo/`).
2. On zenodo.org choose New upload, add all files from `release/zenodo/` (about 6 files plus the checkpoint parts,
   far below the 100 file and 50 GB limits).
3. Paste the fields above, check the SHA-256 values of the files against `MANIFEST.md`, publish.
4. Send the DOI to the agent. It replaces `[[ZENODO_RELEASE_DOI]]` and the author runs `build_paper.py --final`.
