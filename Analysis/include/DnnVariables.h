/*! Definition of DnnMvaVariables.
This file is part of https://github.com/hh-italian-group/hh-bbtautau. */

#pragma once

#include "MvaVariables.h"
#include "AnalysisTools/Core/include/NumericPrimitives.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "tensorflow/core/framework/types.h"
#include <TMatrixD.h>
#include <TMatrixDEigen.h>

namespace analysis {
namespace mva_study{

class DnnMvaVariables : public MvaVariablesBase {
    /*Class for evaluating trained DNN stored in Tensorflow protocol buffer (.pb)*/

    private:
        std::shared_ptr<tensorflow::GraphDef> graphDef;
        std::shared_ptr<tensorflow::Session> session;
        tensorflow::Tensor input(DT_FLOAT, {1, nInputs});
        std::vector<tensorflow::Tensor> outputs;

        //Model config options //Todo: add way of changing this from config file
        int nInputs = 67; 
        bool fixRotate = true;
        std::vector<double> means{1.51975727e+02,  7.52654900e+01,  8.01863353e+01,  6.65015349e-01,
                                5.39720100e-01,  2.16281726e+00,  3.47394597e+02,  1.46177934e+02,
                               -3.43958888e+01,  2.40596679e+02,  1.74137283e+02, -6.14251191e+00,
                                1.22171427e+00,  6.06015127e+01,  2.30166669e+00,  9.02781997e-03,
                                1.12850970e+02,  7.57175906e-01,  2.74846897e-02,  8.09292578e+01,
                               -5.57591636e+00,  2.58883688e+01,  6.02842410e+01,  7.32292869e+01,
                                1.06671012e+02,  5.82524521e+00,  6.00731750e+01,  2.27961314e+00,
                                7.50997040e+01, -2.69267843e-02,  9.86679103e+01,  4.13776832e+02,
                                7.32370268e+01,  6.45864637e-01, -9.15165277e-02,  1.99685160e+02,
                                1.33796957e+01, -5.26061143e-02,  1.02232670e+00,  3.89103972e-02,
                                7.33541062e-01,  9.05896742e-01, -6.45897434e-02, -2.88199642e+01,
                                2.77083011e+00,  2.98825635e-02,  3.76297981e+02,  2.47004615e+02,
                                1.06422319e+00,  8.92801863e+01,  1.12190555e+00,  2.26928485e+00,
                                9.62177250e-01,  2.27978916e+00,  1.72749791e+02,  9.49644769e-01,
                                2.44739771e-01,  1.45322584e-01,  2.16866152e+02,  9.49022483e-01,
                                8.97469292e+01,  1.03336092e-01,  2.25481780e+00,  1.05431104e+00,
                                7.96081071e+00,  2.29148069e-02,  2.62496733e+02};

        std::vector<double> scales{1.84480552e+02, 5.64743918e+01, 1.80491078e+02, 7.21862605e-01,
                               1.08463901e+00, 8.57375541e-01, 2.27266132e+02, 1.31747969e+02,
                               8.95539784e+01, 1.69815686e+02, 8.14561298e+01, 4.03259743e+01,
                               1.29212491e+00, 3.81177567e+01, 9.22236194e-01, 3.47127055e+01,
                               1.18240870e+02, 4.58208898e-01, 5.52398357e+01, 8.40699433e+01,
                               4.26822937e+01, 8.35897838e+01, 9.44174935e+01, 6.35768731e+01,
                               1.35527961e+02, 7.11098780e+01, 4.41501829e+01, 8.35063799e-01,
                               5.24411617e+01, 4.39083518e+01, 7.76575987e+01, 2.01006996e+02,
                               6.35740948e+01, 1.98028092e-01, 7.44401763e+01, 1.89707773e+02,
                               8.84189848e+00, 7.23776750e+01, 3.96959030e-01, 6.88636135e+01,
                               5.35210934e-01, 2.96898577e-01, 7.98386587e+01, 9.37883619e+01,
                               7.89903627e-01, 5.74412042e+01, 2.03471762e+02, 1.20611261e+02,
                               3.81936160e-01, 8.51928247e+01, 3.54703184e-01, 8.34483299e-01,
                               7.14557639e-02, 8.45458138e-01, 1.71230964e+02, 4.14316855e-01,
                               1.06994848e+02, 1.17839679e-01, 2.43664536e+02, 4.96963604e-02,
                               8.51913718e+01, 2.90527885e-02, 8.45313249e-01, 3.86712450e-01,
                               4.46438449e+00, 3.59402079e-02, 2.13738541e+02};

    public:
        DnnMvaVariables(const std::string& model) {
            /*Model = name and location of models to be loaded, without .pb*/

            //Todo: add loading of config file with features, preprop settings, etc.
            graphDef = tensorflow::loadGraphDef(model + ".pb")
            session = tensorflow::createSession(graphDef)
        }

        ~DnnMvaVariables() override {
            /*Close session and delelte model*/
            tensorflow::closeSession(session);
            delete graphDef;
        }

        void getGlobalEventInfo(auto* v_tau_0, auto* v_tau_1, auto* v_bJet_0, auto* v_bJet_1, auto* v_met,
            double*  hT, double*  sT, double* centrality, double* eVis, bool tautau=false) {
            /*Fills referenced variables with global event information*/

            //Reset variables
            *hT = 0;
            *sT = 0;
            *centrality = 0;
            *eVis = 0;
            //HT
            *hT += static_cast<float>(v_bJet_0->Et());
            *hT += static_cast<float>(v_bJet_1->Et());
            *hT += static_cast<float>(v_tau_0->Et());
            if (tautau == true) {
                *hT += static_cast<float>(v_tau_1->Et());
            }
            //ST
            *sT += *hT;
            if (tautau == false) {
                *sT += static_cast<float>(v_tau_1->Pt());
            }
            *sT += v_met->Pt();
            //Centrality
            *eVis += static_cast<float>(v_tau_0->E());
            *centrality += static_cast<float>(v_tau_0->Pt());
            *eVis += static_cast<float>(v_tau_1->E());
            *centrality += static_cast<float>(v_tau_1->Pt());
            *eVis += static_cast<float>(v_bJet_0->E());
            *centrality += static_cast<float>(v_bJet_0->Pt());
            *eVis += static_cast<float>(v_bJet_1->E());
            *centrality += static_cast<float>(v_bJet_1->Pt());
            *centrality /= *eVis;
        }

        TMatrixD decomposeVector(auto* in) {
            TMatrixD out(3, 3);
            out(0, 0) = static_cast<float>(in->Px())*static_cast<float>(in->Px());
            out(0, 1) = static_cast<float>(in->Px())*static_cast<float>(in->Py());
            out(0, 2) = static_cast<float>(in->Px())*static_cast<float>(in->Pz());
            out(1, 0) = static_cast<float>(in->Py())*static_cast<float>(in->Px());
            out(1, 1) = static_cast<float>(in->Py())*static_cast<float>(in->Py());
            out(1, 2) = static_cast<float>(in->Py())*static_cast<float>(in->Pz());
            out(2, 0) = static_cast<float>(in->Pz())*static_cast<float>(in->Px());
            out(2, 1) = static_cast<float>(in->Pz())*static_cast<float>(in->Py());
            out(2, 2) = static_cast<float>(in->Pz())*static_cast<float>(in->Pz());
            return out;
        }

        void appendSphericity(TMatrixD* mat, double* div, TLorentzVector* mom) {
            /*Used in calculating sphericity tensor*/

            TMatrixD decomp = decomposeVector(mom);
            *mat += decomp;
            *div += pow(static_cast<float>(mom->P()), 2);
        }   

        void appendSpherocity(TMatrixD* mat, double* div, TLorentzVector* mom) {
            /*Used in calculating spherocity tensor*/

            TMatrixD decomp = decomposeVector(mom);
            decomp *= 1/std::abs(static_cast<float>(mom->P()));
            *mat += decomp;
            *div += std::abs(static_cast<float>(mom->P()));
        }

        std::vector<double> getEigenValues(TMatrixD in) {
            /*Return vector of sorted, nomalised eigenvalues of passed matrix*/

            TMatrixD eigenMatrix = TMatrixDEigen(in).GetEigenValues();
            std::vector<double> eigenValues(3);
            eigenValues[0] = eigenMatrix(0, 0);
            eigenValues[1] = eigenMatrix(1, 1);
            eigenValues[2] = eigenMatrix(2, 2);
            std::sort(eigenValues.begin(), eigenValues.end(), std::greater<double>());
            double sum = 0;
            for (double n : eigenValues) sum += n;
            std::for_each(eigenValues.begin(), eigenValues.end(), [sum](double i) { return i/sum; });
            return eigenValues;
        }

        void getEventShapes(std::vector<double> sphericityV, std::vector<double> spherocityV,
            double* sphericity, double* spherocity,
            double* aplanarity, double* aplanority,
            double* upsilon, double* dShape) {
            /*Fill referenced features with event shape information*/

            *sphericity = (3/2)*(sphericityV[1]+sphericityV[2]);
            *spherocity = (3/2)*(spherocityV[1]+spherocityV[2]);
            *aplanarity = 3*sphericityV[2]/2;
            *aplanority = 3*spherocityV[2]/2;
            *upsilon = sqrt(3.0)*(sphericityV[1]-sphericityV[2])/2;
            *dShape = 27*spherocityV[0]*spherocityV[1]*spherocityV[2];
        }

        void getPrimaryEventShapes(auto* v_tau_0, auto* v_tau_1,
            auto* v_bJet_0, auto* v_bJet_1,
            double* sphericity, double* spherocity,
            double* aplanarity, double* aplanority,
            double* upsilon, double* dShape,
            double* sphericityEigen0, double* sphericityEigen1, double* sphericityEigen2,
            double* spherocityEigen0, double* spherocityEigen1, double* spherocityEigen2) {
            /*Sets values of referenced event-shape variables for final-states*/

            //Reset values
            *sphericity = 0;
            *spherocity = 0;
            *aplanarity = 0;
            *aplanority = 0;
            *upsilon = 0;
            *dShape = 0;
            *sphericityEigen0 = 0;
            *sphericityEigen1 = 0;
            *sphericityEigen2 = 0;
            *spherocityEigen0 = 0;
            *spherocityEigen1 = 0;
            *spherocityEigen2 = 0;

            //Populate tensors
            TMatrixD sphericityT(3, 3), spherocityT(3, 3);
            double sphericityD = 0, spherocityD = 0;
            appendSphericity(&sphericityT, &sphericityD, v_tau_0);
            appendSpherocity(&spherocityT, &spherocityD, v_tau_0);
            appendSphericity(&sphericityT, &sphericityD, v_tau_1);
            appendSpherocity(&spherocityT, &spherocityD, v_tau_1);
            appendSphericity(&sphericityT, &sphericityD, v_bJet_0);
            appendSpherocity(&spherocityT, &spherocityD, v_bJet_0);
            appendSphericity(&sphericityT, &sphericityD, v_bJet_1);
            appendSpherocity(&spherocityT, &spherocityD, v_bJet_1);
            sphericityT *= 1/sphericityD;
            spherocityT *= 1/spherocityD;

            //Calculate event shapes
            std::vector<double> sphericityV = getEigenValues(sphericityT);
            std::vector<double> spherocityV = getEigenValues(spherocityT);
            getEventShapes(sphericityV, spherocityV,
                sphericity, spherocity,
                aplanarity, aplanority,
                upsilon, dShape);
            *sphericityEigen0 = sphericityV[0];
            *sphericityEigen1 = sphericityV[1];
            *sphericityEigen2 = sphericityV[2];
            *spherocityEigen0 = spherocityV[0];
            *spherocityEigen1 = spherocityV[1];
            *spherocityEigen2 = spherocityV[2];
        }

        void AddEvent(analysis::EventInfoBase& eventbase,
            const SampleId& mass , int spin, double sample_weight = 1., int which_test = -1) override {
            using namespace ROOT::Math::VectorUtil;

            const auto& htt_vis_p4 = eventbase.GetHiggsTTMomentum(false);
            const auto& svFit_p4 = eventbase.GetHiggsTTMomentum(true);
            const auto& t_1_p4 = eventbase.GetLeg(1).GetMomentum(); //Todo: Check ordering
            const auto& t_0_p4 = eventbase.GetLeg(2).GetMomentum();

            const auto& hbb_p4 = eventbase.GetHiggsBB().GetMomentum();
            const auto& bjet0_p4 = eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum();
            const auto& bjet1_p4 = eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum();

            const auto& met_p4 = eventbase.GetMET().GetMomentum();

            //b-jet info
            float b_0_csv = static_cast<float>(eventbase.GetHiggsBB().GetFirstDaughter()->csv());
            float b_0_rawf = static_cast<float>(eventbase.GetHiggsBB().GetFirstDaughter()->rawf());
            float b_0_mva = static_cast<float>(eventbase.GetHiggsBB().GetFirstDaughter()->mva());

            float b_1_csv = static_cast<float>(eventbase.GetHiggsBB().GetSecondDaughter()->csv());
            float b_1_rawf = static_cast<float>(eventbase.GetHiggsBB().GetSecondDaughter()->rawf());
            float b_1_mva = static_cast<float>(eventbase.GetHiggsBB().GetSecondDaughter()->mva());

            //Order jets by pT
            if (bjet0_p4.Pt() < bjet1_p4.Pt()) {
                bjet0_p4 = eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum();
                b_0_csv = static_cast<float>(eventbase.GetHiggsBB().GetSecondDaughter()->csv());
                b_0_rawf = static_cast<float>(eventbase.GetHiggsBB().GetSecondDaughter()->rawf());
                b_0_mva = static_cast<float>(eventbase.GetHiggsBB().GetSecondDaughter()->mva());

                bjet1_p4 = eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum();
                b_1_csv = static_cast<float>(eventbase.GetHiggsBB().GetFirstDaughter()->csv());
                b_1_rawf = static_cast<float>(eventbase.GetHiggsBB().GetFirstDaughter()->rawf());
                b_1_mva = static_cast<float>(eventbase.GetHiggsBB().GetFirstDaughter()->mva());
            }

            //Rotate event to have t_0 at phi=0
            if (fixRotate) {
                t_0_p4.SetPhi(DeltaPhi(t_1_p4, t_0_p4));
                bjet0_p4.SetPhi(DeltaPhi(t_1_p4, bjet0_p4));
                bjet1_p4.SetPhi(DeltaPhi(t_1_p4, bjet1_p4));
                met_p4.SetPhi(DeltaPhi(t_1_p4, met_p4));
                svFit_p4.SetPhi(DeltaPhi(t_1_p4, svFit_p4));
                t_1_p4.SetPhi(0);
            }

            //MET
            float met_px = static_cast<float>(met_p4.Px());
            float met_py = static_cast<float>(met_p4.Py());
            float met_pT = static_cast<float>(met_p4.Pt());

            //Taus
            float t_0_px = static_cast<float>(t_0_p4.Px());
            float t_0_py = static_cast<float>(t_0_p4.Py());
            float t_0_pz = static_cast<float>(t_0_p4.Pz());
            float t_0_P = static_cast<float>(t_0_p4.P());
            float t_0_E = static_cast<float>(t_0_p4.E());
            float t_0_mass = static_cast<float>(t_0_p4.M());
            float t_0_mT = static_cast<float>(Calculate_MT(t_0_p4, met_p4));

            float t_1_px = static_cast<float>(t_1_p4.Px());
            float t_1_py = static_cast<float>(t_1_p4.Py());
            float t_1_pz = static_cast<float>(t_1_p4.Pz());
            float t_1_P = static_cast<float>(t_1_p4.P());
            float t_1_E = static_cast<float>(t_1_p4.E());
            float t_1_mass = static_cast<float>(t_1_p4.M());
            float t_1_mT = static_cast<float>(Calculate_MT(t_1_p4, met_p4));

            //Jets
            float b_0_px = static_cast<float>(bjet0_p4.Px());
            float b_0_py = static_cast<float>(bjet0_p4.Py());
            float b_0_pz = static_cast<float>(bjet0_p4.Pz());
            float b_0_P = static_cast<float>(bjet0_p4.P());
            float b_0_E = static_cast<float>(bjet0_p4.E());
            float b_0_mass = static_cast<float>(bjet0_p4.M());

            float b_1_px = static_cast<float>(bjet1_p4.Px());
            float b_1_py = static_cast<float>(bjet1_p4.Py());
            float b_1_pz = static_cast<float>(bjet1_p4.Pz());
            float b_1_P = static_cast<float>(bjet1_p4.P());
            float b_1_E = static_cast<float>(bjet1_p4.E());
            float b_1_mass = static_cast<float>(bjet1_p4.M());

            //SVFit
            float h_tt_svFit_px = static_cast<float>(svFit_p4.Px());
            float h_tt_svFit_py = static_cast<float>(svFit_p4.Py());
            float h_tt_svFit_pz = static_cast<float>(svFit_p4.Pz());
            float h_tt_svFit_P = static_cast<float>(svFit_p4.P());
            float h_tt_svFit_E = static_cast<float>(svFit_p4.E());
            float h_tt_svFit_mass = static_cast<float>(svFit_p4.M());
            float h_tt_svFit_mT = static_cast<float>(alculate_MT(Htt_sv, met));

            //KinFit
            float diH_kinFit_mass = static_cast<float>(eventbase.GetKinFitResults().mass);
            float diH_kinFit_chi2 = static_cast<float>(eventbase.GetKinFitResults().chi2);
            float diH_kinFit_conv = static_cast<float>(eventbase.GetKinFitResults().conv);

            //h->bb
            float h_bb_px = static_cast<float>(hbb_p4.Px());
            float h_bb_py = static_cast<float>(hbb_p4.Py());
            float h_bb_pz = static_cast<float>(hbb_p4.Pz());
            float h_bb_P = static_cast<float>(hbb_p4.P());
            float h_bb_E = static_cast<float>(hbb_p4.E());
            float h_bb_mass = static_cast<float>(hbb_p4.M());

            //h->tautau
            auto htt_p4 = htt_vis_p4+met_p4;
            float h_tt_px = static_cast<float>(htt_p4.Px());
            float h_tt_py = static_cast<float>(htt_p4.Py());
            float h_tt_pz = static_cast<float>(htt_p4.Pz());
            float h_tt_P = static_cast<float>(htt_p4.P());
            float h_tt_E = static_cast<float>(htt_p4.E());
            float h_tt_mass = static_cast<float>(htt_p4.M());

            //Di-higgs
            auto hh_p4 = hbb_p4+htt_p4;
            float diH_px = static_cast<float>(hh_p4.Px());
            float diH_py = static_cast<float>(hh_p4.Py());
            float diH_pz = static_cast<float>(hh_p4.Pz());
            float diH_P = static_cast<float>(hh_p4.P());
            float diH_E = static_cast<float>(hh_p4.E());
            float diH_mass = static_cast<float>(hh_p4.M());

            //Shapes__________________________
            float hT, sT, centrality, eVis;
            getGlobalEventInfo(&t_0_p4, &t_1_p4, &bjet0_p4, &bjet0_p4, &met_p4,
                &hT, &sT, &centrality, &eVis);

            float sphericity, spherocity, aplanarity, aplanority, upsilon, dShape,
                sphericityEigen0, sphericityEigen1, sphericityEigen2,
                spherocityEigen0, spherocityEigen1, spherocityEigen2;
            getPrimaryEventShapes(&t_0_p4, &t_1_p4, &bjet0_p4, &bjet0_p4,
                &sphericity, &spherocity,
                &aplanarity, &aplanority,
                &upsilon, &dShape,
                &sphericityEigen0, &sphericityEigen1, &sphericityEigen2,
                &spherocityEigen0, &spherocityEigen1, &spherocityEigen2);

            //Twist___________________________
            float twist_b_0_b_1 = static_cast<float>(atan(std::abs(DeltaPhi(bjet0_p4, bjet1_p4)/(bjet0_p4.Eta()-bjet1_p4.Eta()))));
            float twist_b_0_t_0 = static_cast<float>(atan(std::abs(DeltaPhi(bjet0_p4, t_0_p4)/(bjet0_p4.Eta()-t_0_p4.Eta()))));
            float twist_b_0_t_1 = static_cast<float>(atan(std::abs(DeltaPhi(bjet0_p4, t_1_p4)/(bjet0_p4.Eta()-t_1_p4.Eta()))));
            float twist_b_1_t_0 = static_cast<float>(atan(std::abs(DeltaPhi(bjet1_p4, t_0_p4)/(bjet1_p4.Eta()-t_0_p4.Eta()))));
            float twist_b_1_t_1 = static_cast<float>(atan(std::abs(DeltaPhi(bjet1_p4, t_1_p4)/(bjet1_p4.Eta()-t_1_p4.Eta()))));
            float twist_t_0_t_1 = static_cast<float>(atan(std::abs(DeltaPhi(t_0_p4, t_1_p4)/(t_0_p4.Eta()-t_1_p4.Eta()))));
            float twist_h_bb_h_tt = static_cast<float>(atan(std::abs(DeltaPhi(hbb_p4, htt_p4)/(hbb_p4.Eta()-htt_p4.Eta()))));

            //dR__________________________________
            float dR_b_0_b_1 = static_cast<float>(DeltaR(bjet0_p4, bjet1_p4));
            float dR_b_0_t_0 = static_cast<float>(DeltaR(bjet0_p4, t_0_p4));
            float dR_b_0_t_1 = static_cast<float>(DeltaR(bjet0_p4, t_1_p4));
            float dR_b_1_t_0 = static_cast<float>(DeltaR(bjet1_p4, t_0_p4));
            float dR_b_1_t_1 = static_cast<float>(DeltaR(bjet1_p4, t_1_p4));
            float dR_t_0_t_1 = static_cast<float>(DeltaR(t_0_p4, t_1_p4));
            float dR_h_bb_h_tt = static_cast<float>(DeltaR(hbb_p4, htt_p4));

            //['h_tt_svFit_mass', 't_1_mT', 'diH_kinFit_chi2', 'b_0_csv', 'b_1_csv', 'dR_t_0_t_1', 'diH_kinFit_mass', 'h_bb_mass', 'h_bb_px', 'hT', 'h_tt_mass', 't_0_px', 'diH_kinFit_conv', 't_1_px', 'dR_b_0_b_1', 't_0_py', 'h_tt_svFit_mT', 't_0_mass', 'h_tt_svFit_py', 'h_tt_svFit_px', 'b_1_px', 'diH_px', 'h_tt_px', 't_0_P', 'hT_jets', 'met_px', 't_0_mT', 'dR_b_0_t_0', 'met_pT', 'b_1_py', 't_1_E', 'diH_mass', 't_0_E', 'centrality', 'h_bb_py', 'h_bb_P', 'b_0_mass', 'diH_py', 'twist_t_0_t_1', 'h_tt_py', 'b_1_mva', 'b_0_mva', 'b_0_py', 'b_0_px', 'dR_h_bb_h_tt', 'met_py', 'sT', 'h_tt_E', 'twist_b_0_t_1', 'b_1_P', 'twist_h_bb_h_tt', 'dR_b_1_t_0', 'b_1_rawf', 'dR_b_0_t_1', 'b_0_E', 'twist_b_0_b_1', 'b_1_pz', 'sphericity', 'h_tt_svFit_P', 'b_0_rawf', 'b_1_E', 't_1_mass', 'dR_b_1_t_1', 'twist_b_0_t_0', 'b_1_mass', 'aplanarity', 'h_bb_E']
            
            int i = 0;
            input.matrix<float>()(0, i) = (h_tt_svFit_mass - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (t_1_mT; - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (diH_kinFit_chi2 - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_0_csv - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_1_csv - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (dR_t_0_t_1 - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (diH_kinFit_mass - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (h_bb_mass - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (h_bb_px - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (hT - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (h_tt_mass - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (t_0_px - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (diH_kinFit_conv - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (t_1_px - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (dR_b_0_b_1 - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (t_0_py - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (h_tt_svFit_mT - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (t_0_mass - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (h_tt_svFit_py - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (h_tt_svFit_px - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_1_px - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (diH_px - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (h_tt_px - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (t_0_P - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (hT_jets - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (met_px - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (t_0_mT - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (dR_b_0_t_0 - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (met_pT - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_1_py - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (t_1_E - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (diH_mass - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (t_0_E - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (centrality - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (h_bb_py - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (h_bb_P - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_0_mass - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (diH_py - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (twist_t_0_t_1 - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (h_tt_py - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_1_mva - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_0_mva - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_0_py - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_0_px - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (dR_h_bb_h_tt - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (met_py - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (sT - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (h_tt_E - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (twist_b_0_t_1 - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_1_P - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (twist_h_bb_h_tt - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (dR_b_1_t_0 - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_1_rawf - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (dR_b_0_t_1 - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_0_E - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (twist_b_0_b_1 - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_1_pz - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (sphericity - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (h_tt_svFit_P - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_0_rawf - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_1_E - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (t_1_mass - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (dR_b_1_t_1 - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (twist_b_0_t_0 - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (b_1_mass - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (aplanarity - means[i])/scales[i]; i++;
            input.matrix<float>()(0, i) = (h_bb_E - means[i])/scales[i]; i++;
        }

        double Evaluate() override {
            tensorflow::run(session, { { "input", input } }, { "output" }, &outputs);
            return outputs[0].matrix<double>()(0, 0)
        }

        std::shared_ptr<TMVA::Reader> GetReader() override {
            return nullptr;
        }
};

} //mva_study
} //analysis