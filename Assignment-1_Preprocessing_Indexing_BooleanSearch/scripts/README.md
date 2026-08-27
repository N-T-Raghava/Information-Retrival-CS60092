# Terminal Instructions

Run these commands from the assignment directory:

```powershell
cd Assignment-1_Preprocessing_Indexing_BooleanSearch
```

## Compile

```powershell
g++ -std=c++17 -O2 -Wall -Wextra -pedantic .\src\main.cpp .\src\parser\MaamTeam_Parser.cpp .\src\preprocessor\MaamTeam_Preprocessor.cpp .\src\indexer\MaamTeam_Index.cpp .\src\retriever\MaamTeam_Retrieval.cpp -o .\bin\pa1.exe
```

The executable is created at `bin/pa1.exe`.

## Run Each Operation

Preprocess the Cranfield collection:

```powershell
.\bin\pa1.exe preprocess .\data\cran.all.1400 .\data\stopwords.txt group1
```

Build the inverted index:

```powershell
.\bin\pa1.exe index .\group1_processed.all group1
```

Run a Boolean search:

```powershell
.\bin\pa1.exe search .\group1_cran.index .\data\stopwords.txt "aerodynamic AND wing" group1
```

Run all three operations in one command:

```powershell
.\bin\pa1.exe all .\data\cran.all.1400 .\data\stopwords.txt "aerodynamic AND wing" group1
```

Check that these files were created:

- `group1_processed.all`
- `group1_cran.index`
- `group1_query.result`

Clean generated files:

```powershell
Remove-Item .\group1_processed.all, .\group1_cran.index, .\group1_query.result, .\bin\pa1.exe -Force -ErrorAction SilentlyContinue
```