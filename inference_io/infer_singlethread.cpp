#include "common.h"
#include "alias_table.h"
#include "data_stream.h"
#include "data_block.h"
#include "document.h"
#include "meta.h"
#include "util.h"
#include "model.h"
#include "inferer_singlethread.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <thread>
#include <multiverso/barrier.h>

namespace multiverso {
    namespace lightlda {
        class SimpleMeta: public Meta{
        public:
            void InitTF(){
                tf_.resize(Config::num_vocabs, 1);
            }
        };

        class InferSingleThread {
        public:
            static void Run(int argc, char **argv) {
                Log::ResetLogFile("LightLDA_infer_singlethread." + std::to_string(clock()) + ".log");
                Config::Init(argc, argv);

                SimpleMeta meta;
                meta.InitTF();
                //init model
                auto *model = new LocalModel(&meta);
                model->Init();

                xorshift_rng rng;
                SingleThreadInferer inferer(model);
                uint32_t processed = 0;
                while (true) {
                    std::vector<std::pair<int32_t, int32_t>> words;
                    if (ReadDocumentFromStdin(words))
                        break;

                    std::vector<int32_t> doc_memory;
                    doc_memory.push_back(0);
                    for (auto i:words) {
                        for (auto j = 0; j < i.second; j++) {
                            doc_memory.push_back(i.first);
                            doc_memory.push_back(rng.rand_k(Config::num_topics));
                        }
                    }
                    Document doc(&doc_memory[0], &doc_memory[doc_memory.size() - 1]);

                    inferer.SetDocument(&doc);
                    Inference(inferer);
                    DumpDocTopic(doc, processed++);
                }

                delete model;
            }

        private:
            static void Inference(SingleThreadInferer &inferer) {
                inferer.BeforeIteration();
                for (int32_t i = 0; i < Config::num_iterations; ++i) {
                    inferer.DoIteration(i);
                }
                inferer.EndIteration();
            }

            static int ReadDocumentFromStdin(std::vector<std::pair<int32_t, int32_t>> &words) {
                std::string text;
                std::getline(std::cin, text);

                std::string label;
                std::istringstream ss(text);
                ss >> label;

                while (!ss.eof()) {
                    ss >> label;
                    auto colon = std::find(label.begin(), label.end(), ':');
                    if (colon == label.end())
                        return 1;

                    int32_t word = std::stoi(std::string(label.begin(), colon));
                    int32_t count = std::stoi(std::string(colon+1, label.end()));
                    words.emplace_back(word, count);
                }

                if (words.size() == 0) {
                    return 1;
                }

                return 0;
            }

            static void DumpDocTopic(Document& doc, uint32_t processed) {
                Row<int32_t> doc_topic_counter(0, Format::Sparse, kMaxDocLength);
                doc_topic_counter.Clear();
                doc.GetDocTopicVector(doc_topic_counter);
                Row<int32_t>::iterator iter = doc_topic_counter.Iterator();
                std::string output;
                output = "Topics for " + std::to_string(processed) + ": ";
                while (iter.HasNext()) {
                    output += std::to_string(iter.Key()) + ":" + std::to_string(iter.Value()) + " ";
                    iter.Next();
                }
                output += "\n";
                Log::Info(output.c_str());
            }
        };
    } // namespace lightlda
} // namespace multiverso


int main(int argc, char **argv) {
    multiverso::lightlda::Config::inference = true;
    multiverso::lightlda::InferSingleThread::Run(argc, argv);
    return 0;
}
