/*! Definition of BaseEventAnalyzer class, the base class for event analyzers.
This file is part of https://github.com/hh-italian-group/hh-bbtautau. */

#pragma once

#include "AnalysisTools/Run/include/program_main.h"
#include "EventAnalyzerDataCollection.h"
#include "SampleDescriptorConfigEntryReader.h"
#include "h-tautau/Cuts/include/Btag_2016.h"
#include "h-tautau/Cuts/include/hh_bbtautau_2016.h"
#include "h-tautau/Analysis/include/EventLoader.h"
#include "MvaReader.h"
#include "EventAnalyzerData.h"
#include "AnaTuple.h"
#include "EventAnalyzerCore.h"

namespace analysis {

struct AnalyzerArguments : CoreAnalyzerArguments {
    REQ_ARG(std::string, input);
    REQ_ARG(std::string, output);
    OPT_ARG(unsigned, event_set, 0);
};

template<typename _FirstLeg, typename _SecondLeg>
class BaseEventAnalyzer : public EventAnalyzerCore {
public:
    using FirstLeg = _FirstLeg;
    using SecondLeg = _SecondLeg;
    using Event = ntuple::Event;
    using EventPtr = std::shared_ptr<Event>;
    using EventInfo = ::analysis::EventInfo<FirstLeg, SecondLeg>;

    static constexpr Channel ChannelId() { return ChannelInfo::IdentifyChannel<FirstLeg, SecondLeg>(); }

    static EventCategorySet DetermineEventCategories(EventInfo& event)
    {

        static const std::map<DiscriminatorWP, double> btag_working_points = {
            { DiscriminatorWP::Loose, cuts::btag_2016::CSVv2L },
            { DiscriminatorWP::Medium, cuts::btag_2016::CSVv2M }
        };

        EventCategorySet categories;
        categories.insert(EventCategory::Inclusive());

        const bool is_boosted = event.SelectFatJet(cuts::hh_bbtautau_2016::fatJetID::mass,
                                                   cuts::hh_bbtautau_2016::fatJetID::deltaR_subjet) != nullptr;

        if(event.HasBjetPair()) {
            categories.insert(EventCategory::TwoJets_Inclusive());
            const std::vector<const JetCandidate*> jets = {
                &event.GetHiggsBB().GetFirstDaughter(), &event.GetHiggsBB().GetSecondDaughter(),
            };

            std::map<DiscriminatorWP, size_t> bjet_counts;
            for(const auto& jet : jets) {
                for(const auto& btag_wp : btag_working_points) {
                    if((*jet)->csv() > btag_wp.second)
                        ++bjet_counts[btag_wp.first];
                }
            }
            for(const auto& wp_entry : btag_working_points) {
                categories.emplace(2, bjet_counts[wp_entry.first], wp_entry.first);
                categories.emplace(2, bjet_counts[wp_entry.first], wp_entry.first, is_boosted);
            }
        }
        return categories;
    }

    BaseEventAnalyzer(const AnalyzerArguments& _args) :
        EventAnalyzerCore(_args, ChannelId()), args(_args), anaTupleWriter(args.output(), ChannelId())
    {
        InitializeMvaReader();
    }

    void Run()
    {
        ProcessSamples(ana_setup.signals, "signal");
        ProcessSamples(ana_setup.data, "data");
        ProcessSamples(ana_setup.backgrounds, "background");
        std::cout << "Saving output file..." << std::endl;
    }

protected:
    virtual EventRegion DetermineEventRegion(EventInfo& event, EventCategory eventCategory) = 0;

    void InitializeMvaReader()
    {
        if(!mva_setup.is_initialized()) return;
        for(const auto& method : mva_setup->trainings) {
            const auto& name = method.first;
            const auto& file = method.second;
            const auto& vars = mva_setup->variables.at(name);
            const auto& masses = mva_setup->masses.at(name);
            const auto& mass_range_pair= std::minmax_element(masses.begin(), masses.end());
            const Range<int> mass_range(static_cast<int>(*mass_range_pair.first),
                                        static_cast<int>(*mass_range_pair.second));
            const bool legacy = mva_setup->legacy.count(name);
            const bool legacy_lm = legacy && mva_setup->legacy.at(name) == "lm";
            mva_reader.AddRange(mass_range, name, file, vars, legacy, legacy_lm);
        }
    }

    virtual EventSubCategory DetermineEventSubCategory(EventInfo& event, const EventCategory& category,
                                                       std::map<SelectionCut, double>& mva_scores)
    {
        using namespace cuts::hh_bbtautau_2016::hh_tag;
        using MvaKey = std::tuple<std::string, int, int>;

        EventSubCategory sub_category;
        sub_category.SetCutResult(SelectionCut::mh,
                                  IsInsideMassWindow(event.GetHiggsTT(true).GetMomentum().mass(),
                                                     event.GetHiggsBB().GetMomentum().mass(),
                                                     category.HasBoostConstraint() && category.IsBoosted()));
        sub_category.SetCutResult(SelectionCut::KinematicFitConverged,
                                  event.GetKinFitResults().HasValidMass());

        if(mva_setup.is_initialized()) {

            std::map<MvaKey, double> scores;
            for(const auto& mva_sel : mva_setup->selections) {
                const auto& params = mva_sel.second;
                const MvaKey key{params.name, static_cast<int>(params.mass), params.spin};
                if(!scores.count(key))
                    scores[key] = mva_reader.Evaluate(*event, static_cast<int>(params.mass), params.name, params.spin);
                const double score = scores.at(key);
                const bool pass = score > params.cut;
                sub_category.SetCutResult(mva_sel.first, pass);
                mva_scores[mva_sel.first] = score;
            }
        }

        return sub_category;
    }

    void ProcessSamples(const std::vector<std::string>& sample_names, const std::string& sample_set_name)
    {
        std::cout << "Processing " << sample_set_name << " samples... " << std::endl;
        for(size_t sample_index = 0; sample_index < sample_names.size(); ++sample_index) {
            const std::string& sample_name = sample_names.at(sample_index);
            if(!sample_descriptors.count(sample_name))
                throw exception("Sample '%1%' not found.") % sample_name;
            SampleDescriptor& sample = sample_descriptors.at(sample_name);
            if(sample.sampleType == SampleType::QCD || (sample.channels.size() && !sample.channels.count(ChannelId())))
                continue;
            std::cout << '\t' << sample.name << std::endl;

            std::set<std::string> processed_files;
            for(const auto& sample_wp : sample.working_points) {
                if(!sample_wp.file_path.size() || processed_files.count(sample_wp.file_path)) continue;
                auto file = root_ext::OpenRootFile(tools::FullPath({args.input(), sample_wp.file_path}));
                auto tuple = ntuple::CreateEventTuple(ToString(ChannelId()), file.get(), true,
                                                      ntuple::TreeState::Skimmed);
                auto summary_tuple = ntuple::CreateSummaryTuple("summary", file.get(), true,
                                                                ntuple::TreeState::Skimmed);
                const auto prod_summary = ntuple::MergeSummaryTuple(*summary_tuple);
                ProcessDataSource(sample, sample_wp, tuple, prod_summary);
                processed_files.insert(sample_wp.file_path);
            }
        }
    }

    void ProcessDataSource(const SampleDescriptor& sample, const SampleDescriptor::Point& sample_wp,
                           std::shared_ptr<ntuple::EventTuple> tuple, const ntuple::ProdSummary& prod_summary)
    {
        const bool is_signal = ana_setup.IsSignal(sample.name);
        const bool need_to_blind = args.event_set() && (sample.sampleType == SampleType::TT || is_signal);
        const unsigned event_set = args.event_set(), half_split = prod_summary.n_splits / 2;
        const SummaryInfo summary(prod_summary);
        Event prevFullEvent, *prevFullEventPtr = nullptr;
        for(auto tupleEvent : *tuple) {
            if(ntuple::EventLoader::Load(tupleEvent, prevFullEventPtr).IsFull()) {
                prevFullEvent = tupleEvent;
                prevFullEventPtr = &prevFullEvent;
            }
            if(need_to_blind){
                if((event_set == 1 && tupleEvent.split_id >= half_split) || tupleEvent.split_id < half_split)
                    continue;
                tupleEvent.weight_total *= 2;
            }
            EventInfo event(tupleEvent, ntuple::JetPair{0, 1}, &summary);
            if(!ana_setup.energy_scales.count(event.GetEnergyScale())) continue;

            bbtautau::AnaTupleWriter::DataIdMap dataIds;
            const auto eventCategories = DetermineEventCategories(event);
            for(auto eventCategory : eventCategories) {
                if (!EventCategoriesToProcess().count(eventCategory)) continue;
                const EventRegion eventRegion = DetermineEventRegion(event, eventCategory);
                for(const auto& region : EventRegionsToProcess()){
                    if(!eventRegion.Implies(region)) continue;

                    std::map<SelectionCut, double> mva_scores;
                    const auto eventSubCategory = DetermineEventSubCategory(event, eventCategory, mva_scores);
                    for(const auto& subCategory : EventSubCategoriesToProcess()) {
                        if(!eventSubCategory.Implies(subCategory)) continue;
                        SelectionCut mva_cut;
                        double mva_score = 0;
                        if(subCategory.TryGetLastMvaCut(mva_cut))
                            mva_score = mva_scores.at(mva_cut);
                        event.SetMvaScore(mva_score);
                        const EventAnalyzerDataId anaDataId(eventCategory, subCategory, region,
                                                            event.GetEnergyScale(), sample_wp.full_name);
                        if(sample.sampleType == SampleType::Data) {
                            dataIds[anaDataId] = std::make_tuple(1., mva_score);
                        } else {
                            const double weight = event->weight_total * sample.cross_section * ana_setup.int_lumi
                                                / summary->totalShapeWeight;
                            if(sample.sampleType == SampleType::MC) {
                                dataIds[anaDataId] = std::make_tuple(weight, mva_score);
                            } else
                                ProcessSpecialEvent(sample, sample_wp, anaDataId, event, weight, dataIds);
                        }
                        anaTupleWriter.AddEvent(event, dataIds);
                    }
                }
            }
        }
    }

    virtual void ProcessSpecialEvent(const SampleDescriptor& sample, const SampleDescriptor::Point& /*sample_wp*/,
                                     const EventAnalyzerDataId& anaDataId, EventInfo& event, double weight,
                                     bbtautau::AnaTupleWriter::DataIdMap& dataIds)
    {
        if(sample.sampleType == SampleType::DY) {
            static auto const find_b_index = [&]() {
                const auto& param_names = sample.GetModelParameterNames();
                const auto b_param_iter = param_names.find("b");
                if(b_param_iter == param_names.end())
                    throw exception("Unable to find b_parton WP for DY sample");
                return b_param_iter->second;
            };
            static const size_t b_index = find_b_index();

            bool wp_found = false;
            for(const auto& sample_wp : sample.working_points) {
                const size_t n_b_partons = static_cast<size_t>(sample_wp.param_values.at(b_index));
                if(event->jets_nTotal_hadronFlavour_b == n_b_partons ||
                        (n_b_partons == sample.GetNWorkingPoints() - 1
                         && event->jets_nTotal_hadronFlavour_b > n_b_partons)) {
                    const auto finalId = anaDataId.Set(sample_wp.full_name);
                    dataIds[finalId] = std::make_tuple(weight * sample_wp.norm_sf, event.GetMvaScore());
                    wp_found = true;
                    break;
                }
            }
            if(!wp_found)
                throw exception("Unable to find WP for DY event with lhe_n_b_partons = %1%") % event->lhe_n_b_partons;

        } else if(sample.sampleType == SampleType::TT) {
            dataIds[anaDataId] = std::make_tuple(weight / event->weight_top_pt, event.GetMvaScore());
            if(anaDataId.Get<EventEnergyScale>() == EventEnergyScale::Central) {
//                const double weight_topPt = event->weight_total * sample.cross_section * ana_setup.int_lumi
//                        / event.GetSummaryInfo()->totalShapeWeight_withTopPt;
                // FIXME
                dataIds[anaDataId.Set(EventEnergyScale::TopPtUp)] = std::make_tuple(weight, event.GetMvaScore());
                dataIds[anaDataId.Set(EventEnergyScale::TopPtDown)] = std::make_tuple(weight, event.GetMvaScore());
            }
        } else
            throw exception("Unsupported special event type '%1%'.") % sample.sampleType;
    }

protected:
    AnalyzerArguments args;
    bbtautau::AnaTupleWriter anaTupleWriter;
    mva_study::MvaReader mva_reader;
};

} // namespace analysis
