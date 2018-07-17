/*!
 * \file inferer.h
 * \brief data inference
 */
#ifndef LIGHTLDA_INFERER_H_
#define LIGHTLDA_INFERER_H_

// #include <pthread.h>
#include <multiverso/multiverso.h>
#include <multiverso/log.h>
#include <multiverso/barrier.h>
#include <document.h>
#include <meta.h>

namespace multiverso 
{ 
    class Barrier;

namespace lightlda
{
    class AliasTable;
    class LDADataBlock;
    class LightDocSampler;
    class Meta;
    class LocalModel;
    class IDataStream;
    
    class SingleThreadInferer
    {
    public:
        explicit SingleThreadInferer(LocalModel * model);

        ~SingleThreadInferer();
        void SetDocument(Document *doc);
        void BeforeIteration();
        void DoIteration(int32_t iter);
        void EndIteration();
    private:
        std::vector<int32_t> vocab_;
        AliasTableIndex* alias_index_;
        AliasTable* alias_;
        Document* doc_;
        LocalModel * model_;
        LightDocSampler* sampler_;
    };

    inline void SingleThreadInferer::SetDocument(Document *doc) {doc_ = doc;}
} // namespace lightlda
} // namespace multiverso


 #endif //LIGHTLDA_INFERER_H_
