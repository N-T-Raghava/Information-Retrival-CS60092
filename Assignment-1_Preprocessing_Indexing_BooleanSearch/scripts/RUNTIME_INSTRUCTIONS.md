# Terminal Instructions

Run these commands from the assignment directory:

```powershell
cd Assignment-1_Preprocessing_Indexing_BooleanSearch
```

## Compile

```powershell
g++ -std=c++17 -O2 -Wall -Wextra -pedantic .\src\main.cpp .\src\parser\MaamTeam_Parser.cpp .\src\preprocessor\MaamTeam_Preprocessor.cpp .\src\indexer\MaamTeam_Index.cpp .\src\retriever\MaamTeam_Retrieval.cpp -o .\bin\MaamTeam_pa1.exe
```

The executable is created at `bin/pa1.exe`.

## Run Each Operation

Preprocess the Cranfield collection:

```powershell
.\bin\MaamTeam_pa1.exe preprocess .\data\cran.all.1400 .\data\stopwords.txt MaamTeam
```

Build the inverted index:

```powershell
.\bin\MaamTeam_pa1.exe index .\MaamTeam_processed.all MaamTeam
```

We can see MaamTeam_cran.index and MaamTeam_processed.all in the home directory.

Run a Boolean search:

```powershell
.\bin\MaamTeam_pa1.exe search .\MaamTeam_cran.index .\data\stopwords.txt "aerodynamic AND wing" MaamTeam
```

Check that these files were created:

- `MaamTeam_processed.all`
- `MaamTeam_cran.index`
- `MaamTeam_query.result` (Contains the result of the last run query)

## Testing with Sample Queries

Run the automated verification script to test all 12 sample queries:

```powershell
cd tests
.\verify_queries.ps1
```

Example output:
```
Query 1: 'aeroelastic AND aircraft'
  AND: 5 (expected 5) [PASS] | OR: 84 (expected 84) [PASS]
Query 2: 'dynamics AND effects'
  AND: 34 (expected 34) [PASS] | OR: 585 (expected 586) [PASS]
```

Clean generated files:

```powershell
Remove-Item .\MaamTeam_processed.all, .\MaamTeam_cran.index, .\MaamTeam_query.result, .\bin\MaamTeam_pa1.exe -Force -ErrorAction SilentlyContinue
```