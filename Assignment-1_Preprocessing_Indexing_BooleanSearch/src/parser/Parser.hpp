#pragma once

#include <string>
#include <vector>

namespace pa1 {

// Represents a document from the Cranfield collection
struct Document {
    int id = 0;
    std::string title;
    std::string abstractText;
};

class TextPreprocessor;  // Forward declaration

/**
 * Parses the Cranfield collection (cran.all.1400) format.
 * 
 * Recognizes:
 *   .I  (document ID)
 *   .T  (title - may be multiline)
 *   .A  (author - ignored for indexing)
 *   .B  (bibliography - ignored for indexing)
 *   .W  (abstract/body - may be multiline)
 * 
 * Stops, titles, and abstracts are extracted for later preprocessing.
 */
class CranfieldParser {
public:
    /**
     * Parses the Cranfield collection file.
     * @param path  Path to cran.all.1400
     * @return      Vector of parsed documents
     * @throws      std::runtime_error if file cannot be opened
     */
    std::vector<Document> parse(const std::string& path) const;
};

/**
 * Writes preprocessed documents to a collection file.
 * 
 * Output format:
 *   .I <docid>
 *   .S <processed term 1> <processed term 2> ...
 */
class ProcessedCollectionWriter {
public:
    /**
     * Writes preprocessed documents.
     * @param path           Output file path
     * @param documents      Parsed documents
     * @param preprocessor   TextPreprocessor instance for processing text
     */
    void write(const std::string& path,
               const std::vector<Document>& documents,
               const TextPreprocessor& preprocessor) const;
};

}  // namespace pa1
