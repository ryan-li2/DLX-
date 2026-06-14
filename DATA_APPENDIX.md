# Data Appendix

This appendix describes the datasets included with the DLX+ experiments and
the file conventions used by the Python and C++ implementations.

## Experimental Design

The repository contains 240 CC-MWIS problem instances:

- 3 user-distribution classes;
- 4 user counts per class;
- 20 independently generated samples per class and user count.

| Instance | Distribution |
|---:|---|
| 4 | Realistic |
| 8 | Random |
| 12 | Clustered |

The user counts are `500`, `1000`, `2500`, and `5000`. Sample identifiers
range from `0` to `19`. All reported experiments use a maximum of 60 selected
beams and four colours.

## Directory Structure

```text
Data/
  User Data/
    Instance_4.xlsx
    Instance_8.xlsx
    Instance_12.xlsx
  Beam Candidate/
    Beams_<instance>_<users>_60_4_sample<sample>.csv
    Allocate_<instance>_<users>_60_4_sample<sample>.txt
    Pairs_<instance>_<users>_sample<sample>.json
```

There are three user-data workbooks and 240 files of each beam-candidate file
type.

## User Data

Each workbook contains 5,000 user records. Experiments with fewer users use
the first `INS` rows of the corresponding workbook.

| Column | Meaning |
|---|---|
| `Lng` | User longitude |
| `Lat` | User latitude |
| `Demand` | User demand |
| `Terminal_Diameter` | User-terminal antenna diameter |
| `Efficiency` | Antenna efficiency |
| `Frequency` | Operating frequency |
| `Gain` | Calculated antenna gain |
| `G_T` | Gain-to-noise-temperature ratio |
| `Differential_Lng` | Longitude difference used in link calculations |
| `R_l` | Intermediate geometric quantity |
| `R_z` | Intermediate geometric quantity |
| `Int_Angle` | Intermediate angular quantity |
| `Adjusted_Earth_Radius` | Earth-radius value used by the link model |
| `Slant_Range` | User-to-satellite slant range |
| `Elevation_Angle` | User elevation angle |
| `Intermediate_Angle` | Intermediate angular quantity |
| `User_Index` | Zero-based user identifier |

## Beam Candidate Files

Each problem is represented by three files with matching instance, user-count,
and sample identifiers.

### Beam CSV

`Beams_<instance>_<users>_60_4_sample<sample>.csv` contains one row per
candidate beam:

| Column | Meaning |
|---|---|
| Unnamed first column | Zero-based beam identifier |
| `Beam_Lng` | Beam-centre longitude |
| `Beam_Lat` | Beam-centre latitude |
| `Beamwidth` | Beamwidth used for the candidate |
| `Demand` | Total demand covered by the candidate beam |

The `Demand` column supplies the vertex weight in the CC-MWIS formulation.

### Allocation Text

`Allocate_<instance>_<users>_60_4_sample<sample>.txt` stores a Python
dictionary mapping each beam identifier to the identifiers of its covered
users:

```python
{beam_id: [user_id_1, user_id_2, ...]}
```

The Python interface loads this file using `ast.literal_eval`.

### Conflict-Pair JSON

`Pairs_<instance>_<users>_sample<sample>.json` stores a JSON list of beam
pairs:

```json
[[beam_a, beam_b], [beam_c, beam_d]]
```

Each pair identifies two candidate beams connected by the colouring-conflict
relation used when constructing the CC-MWIS instance.

## Problem Identification

A problem is uniquely identified by:

```text
(Instance, User, Simulation)
```

For example, `(4, 500, 0)` refers to the first 500-user Realistic sample.
These three fields should be used when joining DLX+, Gurobi, CP-SAT, ML, and
LP-boost result files. Results should not be matched using CSV row order.

## Reproducing a Data Load

```python
from dlx_experiment import solve

result = solve(
    data_dir="Data",
    instance=4,
    users=500,
    sample=0,
    threads=10,
)
```

The loader reads the corresponding workbook, beam CSV, allocation text, and
conflict-pair JSON before passing the constructed problem to the C++ solver.

## Scope

The repository provides the processed experimental inputs required to
reproduce the reported solver comparisons. Data provenance and redistribution
terms should be added before public archival if they are governed by an
external source or licence.
