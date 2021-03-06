/*! The sm weight.
This file is part of https://github.com/hh-italian-group/hh-bbtautau. */

#pragma once

#include "AnalysisTools/Core/include/RootExt.h"
#include "h-tautau/Analysis/include/EventTuple.h"
#include "h-tautau/McCorrections/include/WeightProvider.h"

namespace analysis {
namespace mc_corrections {

class HH_BMStoSM_weight : public IWeightProvider {
public:
    using Hist = TH2;
    using HistPtr = std::shared_ptr<Hist>;

    HH_BMStoSM_weight(const std::string& sm_weight_file_name, const std::string& hist_name) :
        sm_weight(LoadSMweight(sm_weight_file_name, hist_name)) { }

    virtual double Get(const ntuple::Event& event) const override { return GetT(event); }
    virtual double Get(const ntuple::ExpressEvent& event) const override { return GetT(event); }

private:
    template<typename Event>
    double GetT(const Event& event) const
    {
        double m_hh = event.lhe_hh_m;
        double cos_Theta = event.lhe_hh_cosTheta;
        const Int_t bin_x = sm_weight->GetXaxis()->FindBin(m_hh);
        const Int_t bin_y = sm_weight->GetYaxis()->FindBin(std::abs(cos_Theta));
        if(bin_x < 1 || bin_x > sm_weight->GetNbinsX() || bin_y < 1 || bin_y > sm_weight->GetNbinsY())
            throw exception("Unable to estimate HH BSM to SM weight for the event with m_hh = %1%"
                            " and cos(theta) = %2%.") % m_hh % cos_Theta;

        return sm_weight->GetBinContent(bin_x,bin_y);
    }

private:
    static HistPtr LoadSMweight(const std::string& sm_weight_file_name, const std::string& hist_name)
    {
        auto inputFile_weight = root_ext::OpenRootFile(sm_weight_file_name);
        return HistPtr(root_ext::ReadCloneObject<Hist>(*inputFile_weight, hist_name, "", true));
    }

private:
    HistPtr sm_weight;
};

} // namespace mc_corrections
} // namespace analysis
