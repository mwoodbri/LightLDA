#include "inferer_singlethread.h"

#include "alias_table.h"
#include "common.h"
#include "data_block.h"
#include "meta.h"
#include "sampler.h"
#include "model.h"
#include "data_stream.h"
#include <multiverso/stop_watch.h>
#include <multiverso/log.h>
#include <multiverso/barrier.h>
#include <algorithm>

namespace multiverso { namespace lightlda
{
    SingleThreadInferer::SingleThreadInferer(LocalModel * model):
        alias_(nullptr), doc_(nullptr),
        model_(model)
    {
        sampler_ = new LightDocSampler();
    }

    SingleThreadInferer::~SingleThreadInferer()
    {
        delete sampler_;
    }

    void SingleThreadInferer::BeforeIteration()
    {
        StopWatch watch; watch.Start();

        vocab_.clear();
        for(int32_t i = 0;i<doc_->Size();++i){
            int32_t word = doc_->Word(i);
            bool found = false;
            for(auto j:vocab_){
                if(j==word) {
                    found = true;
                    break;
                }
            }
            if(!found){
                vocab_.push_back(word);
            }
        }
        std::sort(vocab_.begin(), vocab_.end());

        alias_index_ = new AliasTableIndex();
        int64_t offset = 0;
        for (auto word:vocab_) {
            bool is_dense = false;
            int32_t capacity = Config::num_topics;
            int64_t size = Config::num_topics * 3;
            alias_index_->PushWord(word, is_dense, offset, capacity);
            offset += size;
        }

        Log::Info("Adjusting alias size to %d bytes\n", offset);
        Config::alias_capacity = offset * sizeof(int32_t);

        alias_ = new AliasTable();
        alias_->Init(alias_index_);
        alias_->Build(-1, model_);

        for(auto word:vocab_){
            alias_->Build(word, model_);
        }
        Log::Info("Alias Time used: %.2f s \n", watch.ElapsedSeconds());
    }

    void SingleThreadInferer::DoIteration(int32_t iter)
    {
//        Log::Info("iter=%d\n", iter);
        int32_t lastword = vocab_[vocab_.size()-1];

        sampler_->SampleOneDoc(doc_, 0, lastword, model_, alias_);
    }

    void SingleThreadInferer::EndIteration()
    {
        delete alias_;
        delete alias_index_;
    }

} // namespace lightlda
} // namespace multiverso
