#include "MaamTeam_Preprocessor.hpp"

#include <cctype>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sstream>

namespace pa1 {

namespace {

// Helper: convert to lowercase
static std::string toLower(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

// Helper: check if character is alphanumeric
static bool isAlphaNumeric(char ch) {
    return std::isalnum(static_cast<unsigned char>(ch)) != 0;
}

// Helper: trim whitespace
static std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

// Helper: open file for reading
static void requireFile(const std::string& path, std::ifstream& in) {
    in.open(path);
    if (!in) {
        throw std::runtime_error("Cannot open input file: " + path);
    }
}

/**
 * Porter Stemmer implementation.
 * Encapsulated within this translation unit (not exposed in header).
 */
class PorterStemmer {
private:
    static bool consonant(const std::string& b, int i) {
        const char c = b[i];
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') return false;
        if (c == 'y') return i == 0 ? true : !consonant(b, i - 1);
        return true;
    }

    static int measure(const std::string& b, int j) {
        int n = 0;
        bool seenVowel = false;

        for (int i = 0; i <= j; ++i) {
            if (!consonant(b, i)) {
                seenVowel = true;
            } else if (seenVowel) {
                ++n;
                seenVowel = false;
            }
        }
        return n;
    }

    static bool vowelInStem(const std::string& b, int j) {
        for (int i = 0; i <= j; ++i) {
            if (!consonant(b, i)) return true;
        }
        return false;
    }

    static bool doubleConsonant(const std::string& b, int j) {
        if (j < 1) return false;
        return b[j] == b[j - 1] && consonant(b, j) && consonant(b, j - 1);
    }

    static bool cvc(const std::string& b, int i) {
        if (i < 2 || !consonant(b, i) || consonant(b, i - 1) || !consonant(b, i - 2)) {
            return false;
        }
        const char c = b[i];
        return c != 'w' && c != 'x' && c != 'y';
    }

    static bool endsWith(const std::string& b, const std::string& suffix) {
        return b.size() >= suffix.size() &&
               b.compare(b.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    static void replaceSuffix(std::string& b, const std::string& suffix,
                              const std::string& replacement) {
        b.replace(b.size() - suffix.size(), suffix.size(), replacement);
    }

    static bool step1a(std::string& b) {
        if (endsWith(b, "sses")) {
            replaceSuffix(b, "sses", "ss");
        } else if (endsWith(b, "ies")) {
            replaceSuffix(b, "ies", "i");
        } else if (endsWith(b, "ss")) {
            // unchanged
        } else if (endsWith(b, "s")) {
            b.pop_back();
        }
        return true;
    }

    static bool step1b(std::string& b) {
        if (endsWith(b, "eed")) {
            const int stemEnd = static_cast<int>(b.size()) - 4;
            if (stemEnd >= 0 && measure(b, stemEnd) > 0) {
                b.replace(b.size() - 3, 3, "ee");
            }
            return true;
        }

        bool changed = false;
        int stemEnd = -1;

        if (endsWith(b, "ed")) {
            stemEnd = static_cast<int>(b.size()) - 3;
            if (stemEnd >= 0 && vowelInStem(b, stemEnd)) {
                b.erase(b.size() - 2);
                changed = true;
            }
        } else if (endsWith(b, "ing")) {
            stemEnd = static_cast<int>(b.size()) - 4;
            if (stemEnd >= 0 && vowelInStem(b, stemEnd)) {
                b.erase(b.size() - 3);
                changed = true;
            }
        }

        if (!changed) return true;

        if (endsWith(b, "at")) {
            b += "e";
        } else if (endsWith(b, "bl")) {
            b += "e";
        } else if (endsWith(b, "iz")) {
            b += "e";
        } else if (doubleConsonant(b, static_cast<int>(b.size()) - 1)) {
            const char c = b.back();
            if (c == 'l' || c == 's' || c == 'z') b += 'e';
        } else if (cvc(b, static_cast<int>(b.size()) - 1)) {
            b += 'e';
        }
        return true;
    }

    static bool step1c(std::string& b) {
        if (b.size() > 2 && b.back() == 'y') {
            const int j = static_cast<int>(b.size()) - 2;
            if (vowelInStem(b, j)) b.back() = 'i';
        }
        return true;
    }

    static bool step2(std::string& b) {
        static const std::pair<const char*, const char*> rules[] = {
            {"ational", "ate"}, {"tional", "tion"}, {"enci", "ence"},
            {"anci", "ance"}, {"izer", "ize"}, {"bli", "ble"},
            {"alli", "al"}, {"entli", "ent"}, {"eli", "e"},
            {"ousli", "ous"}, {"ization", "ize"}, {"ation", "ate"},
            {"ator", "ate"}, {"alism", "al"}, {"iveness", "ive"},
            {"fulness", "ful"}, {"ousness", "ous"}, {"aliti", "al"},
            {"iviti", "ive"}, {"biliti", "ble"}, {"logi", "log"}
        };

        for (const auto& rule : rules) {
            auto suffix = rule.first;
            auto replacement = rule.second;
            if (endsWith(b, suffix)) {
                const int j = static_cast<int>(b.size() - std::strlen(suffix) - 1);
                if (j >= 0 && measure(b, j) > 0) {
                    replaceSuffix(b, suffix, replacement);
                }
                return true;
            }
        }
        return true;
    }

    static bool step3(std::string& b) {
        static const std::pair<const char*, const char*> rules[] = {
            {"icate", "ic"}, {"ative", ""}, {"alize", "al"},
            {"iciti", "ic"}, {"ical", "ic"}, {"ful", ""}, {"ness", ""}
        };

        for (const auto& rule : rules) {
            auto suffix = rule.first;
            auto replacement = rule.second;
            if (endsWith(b, suffix)) {
                const int j = static_cast<int>(b.size() - std::strlen(suffix) - 1);
                if (j >= 0 && measure(b, j) > 0) {
                    replaceSuffix(b, suffix, replacement);
                }
                return true;
            }
        }
        return true;
    }

    static bool step4(std::string& b) {
        static const char* suffixes[] = {
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
            "ous", "ive", "ize"
        };

        for (const char* suffix : suffixes) {
            if (!endsWith(b, suffix)) continue;

            const int j = static_cast<int>(b.size() - std::strlen(suffix) - 1);
            if (j < 0 || measure(b, j) <= 1) return true;

            if (std::string(suffix) == "ion") {
                if (j >= 0 && (b[j] == 's' || b[j] == 't')) {
                    b.erase(b.size() - std::strlen(suffix));
                }
            } else {
                b.erase(b.size() - std::strlen(suffix));
            }
            return true;
        }
        return true;
    }

    static bool step5a(std::string& b) {
        if (!endsWith(b, "e")) return true;

        const int j = static_cast<int>(b.size()) - 2;
        if (j < 0) return true;

        const int m = measure(b, j);
        if (m > 1 || (m == 1 && !cvc(b, j))) {
            b.pop_back();
        }
        return true;
    }

    static bool step5b(std::string& b) {
        if (endsWith(b, "ll") && measure(b, static_cast<int>(b.size()) - 1) > 1) {
            b.pop_back();
        }
        return true;
    }

public:
    std::string stem(std::string word) const {
        if (word.size() <= 2) return word;

        word = toLower(std::move(word));

        for (char c : word) {
            if (!std::isalpha(static_cast<unsigned char>(c))) return word;
        }

        step1a(word);
        step1b(word);
        step1c(word);
        step2(word);
        step3(word);
        step4(word);
        step5a(word);
        step5b(word);
        return word;
    }
};

}  // anonymous namespace

void TextPreprocessor::loadStopWords(const std::string& path) {
    std::ifstream in;
    requireFile(path, in);
    stopWords_.clear();

    std::string word;
    while (in >> word) {
        word = toLower(trim(word));
        if (!word.empty()) stopWords_.insert(word);
    }
}

std::vector<std::string> TextPreprocessor::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current;

    for (char ch : text) {
        if (isAlphaNumeric(ch)) {
            current.push_back(ch);
        } else if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    }

    if (!current.empty()) tokens.push_back(std::move(current));
    return tokens;
}

std::vector<std::string> TextPreprocessor::normalize(const std::vector<std::string>& tokens) const {
    std::vector<std::string> result;
    result.reserve(tokens.size());

    for (const auto& token : tokens) {
        std::string normalized;
        normalized.reserve(token.size());
        for (char ch : token) {
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
        if (!normalized.empty()) result.push_back(std::move(normalized));
    }
    return result;
}

std::vector<std::string> TextPreprocessor::stem(const std::vector<std::string>& tokens) const {
    static const PorterStemmer stemmer;
    std::vector<std::string> result;
    result.reserve(tokens.size());
    for (const auto& token : tokens) {
        result.push_back(stemmer.stem(token));
    }
    return result;
}

std::vector<std::string> TextPreprocessor::removeStopWords(const std::vector<std::string>& tokens) const {
    std::vector<std::string> result;
    result.reserve(tokens.size());
    for (const auto& token : tokens) {
        if (stopWords_.find(token) == stopWords_.end()) {
            result.push_back(token);
        }
    }
    return result;
}

std::vector<std::string> TextPreprocessor::preprocess(const std::string& text) const {
    auto tokens = tokenize(text);
    tokens = normalize(tokens);
    tokens = stem(tokens);
    tokens = removeStopWords(tokens);
    return tokens;
}

}  // namespace pa1
