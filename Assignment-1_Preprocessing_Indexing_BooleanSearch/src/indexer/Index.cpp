#include "Index.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <unordered_set>

namespace pa1 {

namespace {

// Helper: open file for reading
static void requireFile(const std::string& path, std::ifstream& in) {
    in.open(path);
    if (!in) {
        throw std::runtime_error("Cannot open input file: " + path);
    }
}

// Helper: open file for writing
static void requireOutput(const std::string& path, std::ofstream& out) {
    out.open(path);
    if (!out) {
        throw std::runtime_error("Cannot create output file: " + path);
    }
}

// Helper: trim whitespace
static std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

}  // anonymous namespace

void InvertedIndex::addDocument(int docId, const std::vector<std::string>& tokens) {
    maxDocId_ = std::max(maxDocId_, docId);

    std::unordered_set<std::string> uniqueTerms(tokens.begin(), tokens.end());
    for (const auto& term : uniqueTerms) {
        postings_[term].push_back(docId);
    }
}

void InvertedIndex::finalize() {
    for (auto& posting : postings_) {
        auto& postingList = posting.second;
        std::sort(postingList.begin(), postingList.end());
        postingList.erase(std::unique(postingList.begin(), postingList.end()), postingList.end());
    }
}

const InvertedIndex::PostingList* InvertedIndex::lookup(const std::string& term) const {
    auto it = postings_.find(term);
    if (it == postings_.end()) return nullptr;
    return &it->second;
}

void InvertedIndex::write(const std::string& path) const {
    std::ofstream out;
    requireOutput(path, out);

    // Collect and sort vocabulary
    std::vector<std::string> vocabulary;
    vocabulary.reserve(postings_.size());
    for (const auto& posting : postings_) {
        vocabulary.push_back(posting.first);
    }
    std::sort(vocabulary.begin(), vocabulary.end());

    // Write header
    out << vocabulary.size() << ' ' << maxDocId_ << '\n';

    // Write each term and its posting list
    for (const auto& term : vocabulary) {
        out << term << ' ';
        const auto& postingList = postings_.at(term);
        for (std::size_t i = 0; i < postingList.size(); ++i) {
            if (i != 0) out << ',';
            out << postingList[i];
        }
        out << '\n';
    }
}

void InvertedIndex::read(const std::string& path) {
    std::ifstream in;
    requireFile(path, in);

    postings_.clear();
    maxDocId_ = 0;

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("Index file is empty: " + path);
    }

    std::istringstream header(line);
    std::size_t vocabularySize = 0;
    header >> vocabularySize >> maxDocId_;

    if (!header || vocabularySize > 100000000ULL) {
        throw std::runtime_error("Invalid index header.");
    }

    for (std::size_t i = 0; i < vocabularySize; ++i) {
        if (!std::getline(in, line)) {
            throw std::runtime_error("Index ended before all terms were read.");
        }

        const std::size_t separator = line.find(' ');
        if (separator == std::string::npos) {
            throw std::runtime_error("Malformed index line: " + line);
        }

        const std::string term = line.substr(0, separator);
        const std::string postingText = trim(line.substr(separator + 1));

        PostingList postingList;
        std::stringstream ss(postingText);
        std::string id;
        while (std::getline(ss, id, ',')) {
            id = trim(id);
            if (id.empty()) continue;
            try {
                postingList.push_back(std::stoi(id));
            } catch (...) {
                throw std::runtime_error("Invalid docid in index line: " + line);
            }
        }

        std::sort(postingList.begin(), postingList.end());
        postingList.erase(std::unique(postingList.begin(), postingList.end()), postingList.end());
        postings_.emplace(term, std::move(postingList));
    }
}

}  // namespace pa1
