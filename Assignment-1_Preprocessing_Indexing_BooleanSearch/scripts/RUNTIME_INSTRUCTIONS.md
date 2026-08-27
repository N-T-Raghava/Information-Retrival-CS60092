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
.\bin\pa1.exe preprocess .\data\cran.all.1400 .\data\stopwords.txt MaamTeam
```

Build the inverted index:

```powershell
.\bin\pa1.exe index .\MaamTeam_processed.all MaamTeam
```

Run a Boolean search:

```powershell
.\bin\pa1.exe search .\MaamTeam_cran.index .\data\stopwords.txt "aerodynamic AND wing" MaamTeam
```

Run all three operations in one command:

```powershell
.\bin\pa1.exe all .\data\cran.all.1400 .\data\stopwords.txt "aerodynamic AND wing" MaamTeam
```

Check that these files were created:

- `MaamTeam_processed.all`
- `MaamTeam_cran.index`
- `MaamTeam_query.result`

Clean generated files:

```powershell
Remove-Item .\MaamTeam_processed.all, .\MaamTeam_cran.index, .\MaamTeam_query.result, .\bin\MaamTeam_pa1.exe -Force -ErrorAction SilentlyContinue
```