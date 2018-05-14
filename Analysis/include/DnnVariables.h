/*! Definition of DnnMvaVariables.
This file is part of https://github.com/hh-italian-group/hh-bbtautau. */

#pragma once

#include "MvaVariables.h"
#include "AnalysisTools/Core/include/NumericPrimitives.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <TMatrixD.h>
#include <TMatrixDEigen.h>

namespace analysis {
namespace mva_study{

class DnnMvaVariables : public MvaVariablesBase {
    /*Class for evaluating trained DNN stored in Tensorflow protocol buffer (.pb)*/

    private:
        int nInputs; 
        bool fixRotate;
        std::vector<double> means;
        std::vector<double> scales;

        tensorflow::GraphDef* graphDef;  
        tensorflow::Session* session;
        tensorflow::Tensor input;
        std::vector<tensorflow::Tensor> outputs;

    public:
        DnnMvaVariables(const std::string& model) {
            /*Model = name and location of models to be loaded, without .pb*/

            //Todo: add loading of config file
            graphDef = tensorflow::loadGraphDef(model + ".pb");
            session = tensorflow::createSession(graphDef);

            //Model config options //Todo: add way of changing these along with features, preprop settings, etc. from config file
            nInputs = 67; 
            fixRotate = true;
            means = std::vector<double>{1.51975727e+02,  7.52654900e+01,  8.01863353e+01,  6.65015349e-01,
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

            scales = std::vector<double>{1.84480552e+02, 5.64743918e+01, 1.80491078e+02, 7.21862605e-01,
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

            input = tensorflow::Tensor(tensorflow::DT_FLOAT, {1, nInputs});
        }

        ~DnnMvaVariables() override {
            /*Close session and delelte model*/
            tensorflow::closeSession(session);
            delete graphDef;
        }

        void getGlobalEventInfo(TLorentzVector* v_tau_0, TLorentzVector* v_tau_1, TLorentzVector* v_bJet_0, TLorentzVector* v_bJet_1, TLorentzVector* v_met,
            double*  hT, double*  sT, double* centrality, double* eVis, bool tautau=false) {
            /*Fills referenced variables with global event information*/

            //Reset variables
            *hT = 0;
            *sT = 0;
            *centrality = 0;
            *eVis = 0;

            //HT
            *hT += v_bJet_0->Et();
            *hT += v_bJet_1->Et();
            *hT += v_tau_0->Et();
            if (tautau == true) {
                *hT += v_tau_1->Et();
            }

            //ST
            *sT += *hT;
            if (tautau == false) {
                *sT += v_tau_1->Pt();
            }
            *sT += v_met->Pt();

            //Centrality
            *eVis += v_tau_0->E();
            *centrality += v_tau_0->Pt();
            *eVis += v_tau_1->E();
            *centrality += v_tau_1->Pt();
            *eVis += v_bJet_0->E();
            *centrality += v_bJet_0->Pt();
            *eVis += v_bJet_1->E();
            *centrality += v_bJet_1->Pt();
            *centrality /= *eVis;
        }

        TMatrixD decomposeVector(TLorentzVector* in) {
            TMatrixD out(3, 3);
            out(0, 0) = in->Px()*in->Px();
            out(0, 1) = in->Px()*in->Py();
            out(0, 2) = in->Px()*in->Pz();
            out(1, 0) = in->Py()*in->Px();
            out(1, 1) = in->Py()*in->Py();
            out(1, 2) = in->Py()*in->Pz();
            out(2, 0) = in->Pz()*in->Px();
            out(2, 1) = in->Pz()*in->Py();
            out(2, 2) = in->Pz()*in->Pz();
            return out;
        }

        void appendSphericity(TMatrixD* mat, double* div, TLorentzVector* mom) {
            /*Used in calculating sphericity tensor*/

            TMatrixD decomp = decomposeVector(mom);
            *mat += decomp;
            *div += pow(mom->P(), 2);
        }   

        void appendSpherocity(TMatrixD* mat, double* div, TLorentzVector* mom) {
            /*Used in calculating spherocity tensor*/

            TMatrixD decomp = decomposeVector(mom);
            decomp *= 1/std::abs(mom->P());
            *mat += decomp;
            *div += std::abs(mom->P());
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

        void getPrimaryEventShapes(TLorentzVector* v_tau_0, TLorentzVector* v_tau_1,
            TLorentzVector* v_bJet_0, TLorentzVector* v_bJet_1,
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
            /*Load event features into input tensor*/

            using namespace ROOT::Math::VectorUtil;

            TLorentzVector t_0_p4, t_1_p4, bjet0_p4, bjet1_p4, met_p4, svFit_p4;
            t_0_p4.SetPxPyPzE(eventbase.GetLeg(2).GetMomentum().Px(), eventbase.GetLeg(2).GetMomentum().Py(), eventbase.GetLeg(2).GetMomentum().Pz(), eventbase.GetLeg(2).GetMomentum().E()); //Todo: Check ordering
            t_1_p4.SetPxPyPzE(eventbase.GetLeg(1).GetMomentum().Px(), eventbase.GetLeg(1).GetMomentum().Py(), eventbase.GetLeg(1).GetMomentum().Pz(), eventbase.GetLeg(1).GetMomentum().E());
            bjet0_p4.SetPxPyPzE(eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().Px(), eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().Py(), eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().Pz(), eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().E());
            bjet1_p4.SetPxPyPzE(eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().Px(), eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().Py(), eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().Pz(), eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().E());
            met_p4.SetPxPyPzE(eventbase.GetMET().GetMomentum().Px(), eventbase.GetMET().GetMomentum().Py(), eventbase.GetMET().GetMomentum().Pz(), eventbase.GetMET().GetMomentum().E());
            svFit_p4.SetPxPyPzE(eventbase.GetHiggsTTMomentum(true).Px(), eventbase.GetHiggsTTMomentum(true).Py(), eventbase.GetHiggsTTMomentum(true).Pz(), eventbase.GetHiggsTTMomentum(true).E());

            //b-jet info
            double b_0_csv = static_cast<double>(eventbase.GetHiggsBB().GetFirstDaughter()->csv());
            double b_0_rawf = static_cast<double>(eventbase.GetHiggsBB().GetFirstDaughter()->rawf());
            double b_0_mva = static_cast<double>(eventbase.GetHiggsBB().GetFirstDaughter()->mva());

            double b_1_csv = static_cast<double>(eventbase.GetHiggsBB().GetSecondDaughter()->csv());
            double b_1_rawf = static_cast<double>(eventbase.GetHiggsBB().GetSecondDaughter()->rawf());
            double b_1_mva = static_cast<double>(eventbase.GetHiggsBB().GetSecondDaughter()->mva());

            //Order jets by pT
            if (bjet0_p4.Pt() < bjet1_p4.Pt()) {
                bjet0_p4.SetPxPyPzE(eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().Px(), eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().Py(), eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().Pz(), eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().E());
                b_0_csv = static_cast<double>(eventbase.GetHiggsBB().GetSecondDaughter()->csv());
                b_0_rawf = static_cast<double>(eventbase.GetHiggsBB().GetSecondDaughter()->rawf());
                b_0_mva = static_cast<double>(eventbase.GetHiggsBB().GetSecondDaughter()->mva());

                bjet1_p4.SetPxPyPzE(eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().Px(), eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().Py(), eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().Pz(), eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().E());
                b_1_csv = static_cast<double>(eventbase.GetHiggsBB().GetFirstDaughter()->csv());
                b_1_rawf = static_cast<double>(eventbase.GetHiggsBB().GetFirstDaughter()->rawf());
                b_1_mva = static_cast<double>(eventbase.GetHiggsBB().GetFirstDaughter()->mva());
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

            TLorentzVector hbb_p4 = bjet0_p4+bjet1_p4;
            TLorentzVector htt_p4 = t_0_p4+t_1_p4+met_p4;
            TLorentzVector hh_p4 = hbb_p4+htt_p4;

            //Global info
            double nJets = static_cast<double>(eventbase.GetNJets());
            double hT_jets = 0;
            for (const JetCandidate& jet : eventbase.GetJets()) {
                hT_jets += jet.GetMomentum().Et()
            }

            //MET
            double met_px = met_p4.Px();
            double met_py = met_p4.Py();
            double met_pT = met_p4.Pt();

            //Taus
            double t_0_px = t_0_p4.Px();
            double t_0_py = t_0_p4.Py();
            double t_0_pz = t_0_p4.Pz();
            double t_0_P = t_0_p4.P();
            double t_0_E = t_0_p4.E();
            double t_0_mass = t_0_p4.M();
            double t_0_mT = Calculate_MT(t_0_p4, met_p4);

            double t_1_px = t_1_p4.Px();
            double t_1_py = t_1_p4.Py();
            double t_1_pz = t_1_p4.Pz();
            double t_1_P = t_1_p4.P();
            double t_1_E = t_1_p4.E();
            double t_1_mass = t_1_p4.M();
            double t_1_mT = Calculate_MT(t_1_p4, met_p4);

            //Jets
            double b_0_px = bjet0_p4.Px();
            double b_0_py = bjet0_p4.Py();
            double b_0_pz = bjet0_p4.Pz();
            double b_0_P = bjet0_p4.P();
            double b_0_E = bjet0_p4.E();
            double b_0_mass = bjet0_p4.M();

            double b_1_px = bjet1_p4.Px();
            double b_1_py = bjet1_p4.Py();
            double b_1_pz = bjet1_p4.Pz();
            double b_1_P = bjet1_p4.P();
            double b_1_E = bjet1_p4.E();
            double b_1_mass = bjet1_p4.M();

            //SVFit
            double h_tt_svFit_px = svFit_p4.Px();
            double h_tt_svFit_py = svFit_p4.Py();
            double h_tt_svFit_pz = svFit_p4.Pz();
            double h_tt_svFit_P = svFit_p4.P();
            double h_tt_svFit_E = svFit_p4.E();
            double h_tt_svFit_mass = svFit_p4.M();
            double h_tt_svFit_mT = Calculate_MT(eventbase.GetHiggsTTMomentum(true), eventbase.GetMET().GetMomentum());

            //KinFit
            double diH_kinFit_mass = static_cast<double>(eventbase.GetKinFitResults().mass);
            double diH_kinFit_chi2 = static_cast<double>(eventbase.GetKinFitResults().chi2);
            double diH_kinFit_conv = static_cast<double>(eventbase.GetKinFitResults().convergence);

            //h->bb
            double h_bb_px = hbb_p4.Px();
            double h_bb_py = hbb_p4.Py();
            double h_bb_pz = hbb_p4.Pz();
            double h_bb_P = hbb_p4.P();
            double h_bb_E = hbb_p4.E();
            double h_bb_mass = hbb_p4.M();

            //h->tautau
            double h_tt_px = htt_p4.Px();
            double h_tt_py = htt_p4.Py();
            double h_tt_pz = htt_p4.Pz();
            double h_tt_P = htt_p4.P();
            double h_tt_E = htt_p4.E();
            double h_tt_mass = htt_p4.M();

            //Di-higgs
            double diH_px = hh_p4.Px();
            double diH_py = hh_p4.Py();
            double diH_pz = hh_p4.Pz();
            double diH_P = hh_p4.P();
            double diH_E = hh_p4.E();
            double diH_mass = hh_p4.M();

            //Shapes__________________________
            double hT, sT, centrality, eVis;
            getGlobalEventInfo(&t_0_p4, &t_1_p4, &bjet0_p4, &bjet0_p4, &met_p4,
                &hT, &sT, &centrality, &eVis);

            double sphericity, spherocity, aplanarity, aplanority, upsilon, dShape,
                sphericityEigen0, sphericityEigen1, sphericityEigen2,
                spherocityEigen0, spherocityEigen1, spherocityEigen2;
            getPrimaryEventShapes(&t_0_p4, &t_1_p4, &bjet0_p4, &bjet0_p4,
                &sphericity, &spherocity,
                &aplanarity, &aplanority,
                &upsilon, &dShape,
                &sphericityEigen0, &sphericityEigen1, &sphericityEigen2,
                &spherocityEigen0, &spherocityEigen1, &spherocityEigen2);

            //Twist___________________________
            double twist_b_0_b_1 = atan(std::abs(DeltaPhi(bjet0_p4, bjet1_p4)/(bjet0_p4.Eta()-bjet1_p4.Eta())));
            double twist_b_0_t_0 = atan(std::abs(DeltaPhi(bjet0_p4, t_0_p4)/(bjet0_p4.Eta()-t_0_p4.Eta())));
            double twist_b_0_t_1 = atan(std::abs(DeltaPhi(bjet0_p4, t_1_p4)/(bjet0_p4.Eta()-t_1_p4.Eta())));
            double twist_b_1_t_0 = atan(std::abs(DeltaPhi(bjet1_p4, t_0_p4)/(bjet1_p4.Eta()-t_0_p4.Eta())));
            double twist_b_1_t_1 = atan(std::abs(DeltaPhi(bjet1_p4, t_1_p4)/(bjet1_p4.Eta()-t_1_p4.Eta())));
            double twist_t_0_t_1 = atan(std::abs(DeltaPhi(t_0_p4, t_1_p4)/(t_0_p4.Eta()-t_1_p4.Eta())));
            double twist_h_bb_h_tt = atan(std::abs(DeltaPhi(hbb_p4, htt_p4)/(hbb_p4.Eta()-htt_p4.Eta())));

            //dR__________________________________
            double dR_b_0_b_1 = DeltaR(bjet0_p4, bjet1_p4);
            double dR_b_0_t_0 = DeltaR(bjet0_p4, t_0_p4);
            double dR_b_0_t_1 = DeltaR(bjet0_p4, t_1_p4);
            double dR_b_1_t_0 = DeltaR(bjet1_p4, t_0_p4);
            double dR_b_1_t_1 = DeltaR(bjet1_p4, t_1_p4);
            double dR_t_0_t_1 = DeltaR(t_0_p4, t_1_p4);
            double dR_h_bb_h_tt = DeltaR(hbb_p4, htt_p4);

            //['h_tt_svFit_mass', 't_1_mT', 'diH_kinFit_chi2', 'b_0_csv', 'b_1_csv', 'dR_t_0_t_1', 'diH_kinFit_mass', 'h_bb_mass', 'h_bb_px', 'hT', 'h_tt_mass', 't_0_px', 'diH_kinFit_conv', 't_1_px', 'dR_b_0_b_1', 't_0_py', 'h_tt_svFit_mT', 't_0_mass', 'h_tt_svFit_py', 'h_tt_svFit_px', 'b_1_px', 'diH_px', 'h_tt_px', 't_0_P', 'hT_jets', 'met_px', 't_0_mT', 'dR_b_0_t_0', 'met_pT', 'b_1_py', 't_1_E', 'diH_mass', 't_0_E', 'centrality', 'h_bb_py', 'h_bb_P', 'b_0_mass', 'diH_py', 'twist_t_0_t_1', 'h_tt_py', 'b_1_mva', 'b_0_mva', 'b_0_py', 'b_0_px', 'dR_h_bb_h_tt', 'met_py', 'sT', 'h_tt_E', 'twist_b_0_t_1', 'b_1_P', 'twist_h_bb_h_tt', 'dR_b_1_t_0', 'b_1_rawf', 'dR_b_0_t_1', 'b_0_E', 'twist_b_0_b_1', 'b_1_pz', 'sphericity', 'h_tt_svFit_P', 'b_0_rawf', 'b_1_E', 't_1_mass', 'dR_b_1_t_1', 'twist_b_0_t_0', 'b_1_mass', 'aplanarity', 'h_bb_E']
            
            std::size_t i = 0; //Todo: find better way of including features
            input.matrix<float>()(0, i) = static_cast<float>((h_tt_svFit_mass - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((t_1_mT - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((diH_kinFit_chi2 - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_0_csv - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_1_csv - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((dR_t_0_t_1 - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((diH_kinFit_mass - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((h_bb_mass - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((h_bb_px - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((hT - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((h_tt_mass - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((t_0_px - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((diH_kinFit_conv - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((t_1_px - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((dR_b_0_b_1 - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((t_0_py - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((h_tt_svFit_mT - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((t_0_mass - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((h_tt_svFit_py - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((h_tt_svFit_px - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_1_px - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((diH_px - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((h_tt_px - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((t_0_P - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((hT_jets - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((met_px - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((t_0_mT - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((dR_b_0_t_0 - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((met_pT - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_1_py - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((t_1_E - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((diH_mass - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((t_0_E - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((centrality - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((h_bb_py - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((h_bb_P - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_0_mass - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((diH_py - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((twist_t_0_t_1 - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((h_tt_py - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_1_mva - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_0_mva - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_0_py - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_0_px - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((dR_h_bb_h_tt - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((met_py - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((sT - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((h_tt_E - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((twist_b_0_t_1 - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_1_P - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((twist_h_bb_h_tt - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((dR_b_1_t_0 - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_1_rawf - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((dR_b_0_t_1 - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_0_E - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((twist_b_0_b_1 - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_1_pz - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((sphericity - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((h_tt_svFit_P - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_0_rawf - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_1_E - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((t_1_mass - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((dR_b_1_t_1 - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((twist_b_0_t_0 - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((b_1_mass - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((aplanarity - means[i])/scales[i]); i++;
            input.matrix<float>()(0, i) = static_cast<float>((h_bb_E - means[i])/scales[i]); i++;
        }

        double Evaluate() override {
            tensorflow::run(session, { { "input", input } }, { "output" }, &outputs);
            return outputs[0].matrix<double>()(0, 0);
        }

        std::shared_ptr<TMVA::Reader> GetReader() override {
            return nullptr;
        }
};

} //mva_study
} //analysis