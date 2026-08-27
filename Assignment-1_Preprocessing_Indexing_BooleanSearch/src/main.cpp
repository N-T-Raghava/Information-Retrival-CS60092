#include "parser/MaamTeam_Parser.hpp"
#include "preprocessor/MaamTeam_Preprocessor.hpp"
#include "indexer/MaamTeam_Index.hpp"
#include "retriever/MaamTeam_Retrieval.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

namespace pa1 {

// Helper: open file for reading
static void requireFile(const std::string& path, std::ifstream& in) {
    in.open(path);
    if (!in) {
        throw std::runtime_error("Cannot open input file: " + path);
    }
}

// Helper: trim whitespace
static std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

// Main application class that orchestrates the preprocessing, indexing, and search pipeline.
class PA1Application {
public:
    // Preprocess: Parse collection and generate processed document file.
    void preprocess(const std::string& collectionPath,
                    const std::string& stopWordsPath,
                    const std::string& groupName) const {
        TextPreprocessor preprocessor;
        preprocessor.loadStopWords(stopWordsPath);

        CranfieldParser parser;
        const auto documents = parser.parse(collectionPath);

        const std::string output = groupName + "_processed.all";
        ProcessedCollectionWriter writer;
        writer.write(output, documents, preprocessor);

        std::cout << "Preprocessing complete.\n"
                  << "Documents: " << documents.size() << '\n'
                  << "Output: " << output << '\n';
    }

    // Index: Build inverted index from preprocessed documents.
    void index(const std::string& processedPath,
               const std::string& groupName) const {
        std::ifstream in;
        requireFile(processedPath, in);

        InvertedIndex index;
        int currentDocId = 0;
        std::string line;

        while (std::getline(in, line)) {
            if (line.rfind(".I", 0) == 0) {
                std::istringstream iss(line.substr(2));
                iss >> currentDocId;
                if (!iss || currentDocId <= 0) {
                    throw std::runtime_error("Invalid .I line: " + line);
                }
            } else if (line.rfind(".S", 0) == 0) {
                if (currentDocId <= 0) {
                    throw std::runtime_error(".S encountered before .I");
                }

                const std::string tokenText = trim(line.substr(2));
                std::istringstream iss(tokenText);
                std::string token;
                std::vector<std::string> tokens;
                while (iss >> token) tokens.push_back(std::move(token));

                index.addDocument(currentDocId, tokens);
            }
        }

        index.finalize();
        const std::string output = groupName + "_cran.index";
        index.write(output);

        std::cout << "Indexing complete.\n"
                  << "Vocabulary: " << index.vocabularySize() << '\n'
                  << "Maximum docid: " << index.maxDocId() << '\n'
                  << "Output: " << output << '\n';
    }

    // Search: Execute Boolean query against index.
    void search(const std::string& indexPath,
                const std::string& stopWordsPath,
                const std::string& query,
                const std::string& groupName) const {
        TextPreprocessor preprocessor;
        preprocessor.loadStopWords(stopWordsPath);

        InvertedIndex index;
        index.read(indexPath);

        QueryProcessor queryProcessor;
        const auto parsed = queryProcessor.parse(query, preprocessor);

        BooleanRetriever retriever;
        const auto result = retriever.execute(index, parsed.left, parsed.op, parsed.right);

        const std::string output = groupName + "_query.result";
        std::ofstream out;
        out.open(output);
        if (!out) {
            throw std::runtime_error("Cannot create output file: " + output);
        }

        for (std::size_t i = 0; i < result.size(); ++i) {
            if (i != 0) out << ' ';
            out << result[i];
        }
        out << '\n';

        std::cout << "Boolean search complete.\n"
                  << "Processed query: " << parsed.left << ' ' << parsed.op << ' ' << parsed.right << '\n'
                  << "Matching documents: " << result.size() << '\n'
                  << "Output: " << output << '\n';
    }

    // All: Execute preprocess -> index -> search pipeline.
    void all(const std::string& collectionPath,
             const std::string& stopWordsPath,
             const std::string& query,
             const std::string& groupName) const {
        preprocess(collectionPath, stopWordsPath, groupName);
        index(groupName + "_processed.all", groupName);
        search(groupName + "_cran.index", stopWordsPath, query, groupName);
    }
};

static void printUsage(const char* executable) {
    std::cerr << "Usage:\n"
              << "  " << executable << " preprocess <cran.all> <stopwords.txt> <group>\n"
              << "  " << executable << " index <group>_processed.all <group>\n"
              << "  " << executable << " search <group>_cran.index <stopwords.txt> \"word AND word\" <group>\n"
              << "  " << executable << " all <cran.all> <stopwords.txt> \"word AND word\" <group>\n";
}

}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            pa1::printUsage(argv[0]);
            return 1;
        }

        const std::string command = argv[1];
        pa1::PA1Application app;

        if (command == "preprocess") {
            if (argc != 5) {
                pa1::printUsage(argv[0]);
                return 1;
            }
            app.preprocess(argv[2], argv[3], argv[4]);
        } else if (command == "index") {
            if (argc != 4) {
                pa1::printUsage(argv[0]);
                return 1;
            }
            app.index(argv[2], argv[3]);
        } else if (command == "search") {
            if (argc != 6) {
                pa1::printUsage(argv[0]);
                return 1;
            }
            app.search(argv[2], argv[3], argv[4], argv[5]);
        } else if (command == "all") {
            if (argc != 6) {
                pa1::printUsage(argv[0]);
                return 1;
            }
            app.all(argv[2], argv[3], argv[4], argv[5]);
        } else {
            std::cerr << "Unknown command: " << command << '\n';
            pa1::printUsage(argv[0]);
            return 1;
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n';
        return 1;
    }
}
