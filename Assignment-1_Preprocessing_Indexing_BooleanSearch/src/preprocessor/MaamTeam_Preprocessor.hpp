#pragma once

#include <string>
#include <unordered_set>
#include <vector>

namespace pa1 {

/**
 * Preprocesses text by tokenizing, normalizing, stemming, and removing stopwords.
 * 
 * Used for both document text and query text.
 */
class TextPreprocessor {
public:
    /**
     * Loads stopwords from file (one per line).
     * @param path  Path to stopwords.txt
     * @throws      std::runtime_error if file cannot be opened
     */
    void loadStopWords(const std::string& path);

    /**
     * Preprocesses raw text: tokenize -> normalize -> stem -> remove stopwords.
     * @param text  Raw text to preprocess
     * @return      Processed tokens (stemmed, lowercased, stopwords removed)
     */
    std::vector<std::string> preprocess(const std::string& text) const;

    /**
     * Tokenizes text by alphanumeric boundaries.
     * @param text  Raw text
     * @return      Vector of tokens
     */
    std::vector<std::string> tokenize(const std::string& text) const;

    /**
     * Normalizes tokens to lowercase.
     * @param tokens  Raw tokens
     * @return        Normalized (lowercased) tokens
     */
    std::vector<std::string> normalize(const std::vector<std::string>& tokens) const;

    /**
     * Removes stopwords from tokens.
     * @param tokens  Input tokens
     * @return        Tokens with stopwords removed
     */
    std::vector<std::string> removeStopWords(const std::vector<std::string>& tokens) const;

    /**
     * Stems tokens using Porter Stemmer algorithm.
     * @param tokens  Input tokens
     * @return        Stemmed tokens
     */
    std::vector<std::string> stem(const std::vector<std::string>& tokens) const;

private:
    std::unordered_set<std::string> stopWords_;
};

}  // namespace pa1
