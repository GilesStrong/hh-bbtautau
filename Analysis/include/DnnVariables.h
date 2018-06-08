/*! Definition of DnnMvaVariables.
This file is part of https://github.com/hh-italian-group/hh-bbtautau. */

#pragma once

#include "MvaVariables.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "TMatrixD.h"
#include "TMatrixT.h"
#include "TMatrixDEigen.h"

namespace analysis {
namespace mva_study{

class DnnMvaVariables : public MvaVariablesBase {
    /*Class for evaluating trained DNN stored in Tensorflow protocol buffer (.pb)*/

    private:
        size_t nInputs; 
        bool fixRotate;
        std::vector<double> means;
        std::vector<double> scales;
        std::map<std::string, double> features;
        std::vector<std::string> inputFeatures;

        tensorflow::GraphDef* graphDef;  
        tensorflow::Session* session;
        tensorflow::Tensor input;
        
        bool debug = false;

    public:
        DnnMvaVariables(const std::string& model) {
            /*Model = name and location of models to be loaded, without .pb*/

            if (debug) std::cout << "Initialising DNN class\n";

            //Todo: add loading of config file
            if (debug) std::cout << "Loading model:" << model + ".pb" << "\n";
            graphDef = tensorflow::loadGraphDef(model + ".pb");
            if (debug) std::cout << "Model loaded, beginning TF session\n";
            session = tensorflow::createSession(graphDef);
            if (debug) std::cout << "Begun TF session\n";

            //Model config options //Todo: add way of changing these along with features, preprop settings, etc. from config file
            inputFeatures = std::vector<std::string>{'t_0_px', 't_0_py', 't_0_pz', 't_0_mass', 't_0_mT', 't_0_mT2', 't_1_px', 't_1_py', 't_1_pz', 't_1_mT', 't_1_mT2', 'b_0_px', 'b_0_py', 'b_0_pz', 'b_0_mass', 'b_0_csv', 'b_0_mva', 'b_1_px', 'b_1_py', 'b_1_pz', 'b_1_mass', 'b_1_csv', 'b_1_mva', 'met_px', 'met_py', 'met_pT', 'h_tt_svFit_px', 'h_tt_svFit_py', 'h_tt_svFit_pz', 'h_tt_svFit_mass', 'h_tt_svFit_mT', 'h_bb_px', 'h_bb_py', 'h_bb_pz', 'h_bb_mass', 'diH_px', 'diH_py', 'diH_pz', 'diH_kinFit_mass', 'diH_kinFit_chi2', 'diH_kinFit_conv', 'hT', 'hT_jets'};
            nInputs = inputFeatures.size();
            if (debug) std::cout << "Number of inputs = " << nInputs << "\n"; 
            fixRotate = false;
            means = std::vector<double>{-0.1657936073143518, -0.02343597761335125, -0.02333487075510067, 0.7571759061763041, 60.07317499459512, 2869.6694468305054, -0.07297886718953847, -0.1424106155513016, 0.44524331472619844, 75.26549004632871, 5125.517629839229, 0.3217521762395806, 0.03167422331295357, 0.1820295551516969, 13.3796957199172, 0.6650153490225249, 0.9058967415839231, 0.19548353148454786, 0.05472876784127139, 0.24473977070067915, 7.960810706709419, 0.5397200998473382, 0.733541062155672, -4.4959444784521425, 0.09154621832555666, 75.09970403638704, -2.058751842006953, -0.1639018615142105, 0.896506130083173, 151.97572651073563, 112.85097035145368, 0.5172357077241284, 0.08640299115422495, 0.42676932585237604, 146.17793435124494, -4.217483141339399, 0.0121026308756812, 0.8486774721410685, 347.39459744227133, 80.18633531111797, 1.2217142696209462, 240.59667854773232, 106.67101155276941};
            scales = std::vector<double>{37.91032026787033, 37.83738043784793, 80.84338157047861, 0.4582088980672853, 44.150182870073564, 9008.729988183684, 50.81346334034539, 50.43284568441607, 103.15202297458833, 56.474391839078265, 10658.757901369183, 89.37596426490317, 89.51476724366489, 207.13346769052075, 8.841898478334382, 0.7218626046045415, 0.29689857740639586, 43.47660391395152, 43.480534550136625, 106.99484782426441, 4.464384494233613, 1.084639013069971, 0.5352109338151005, 64.72501554050605, 64.65671821966762, 52.44116169053776, 91.51396964739864, 91.04765128594751, 299.5560126044877, 184.48055241339125, 118.24086992369644, 85.9020495062545, 85.81902060076776, 247.2219239620548, 131.74796903083555, 80.54171752401825, 79.9451869105092, 328.6573063237755, 227.26613150940463, 180.49107816419902, 1.2921249079276635, 169.81568592889175, 135.52796067601696};

            input = tensorflow::Tensor(tensorflow::DT_FLOAT, {1, static_cast<int>(nInputs)});
            if (debug) std::cout << "DNN class initialised\n";
        }

        ~DnnMvaVariables() override {
            /*Close session and delete model*/
            if (debug) std::cout << "Deleting DNN class\n";
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

        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wundefined-func-template" //CLang complains about TMatrixT not being explicitly defined here, not sure why.
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
        #pragma clang diagnostic pop

        void AddEvent(analysis::EventInfoBase& eventbase,
            const SampleId& /*mass*/ , int /*spin*/, double /*sample_weight = 1.*/, int /*which_test = -1*/) override {
            /*Load event features into input tensor*/
            using namespace ROOT::Math::VectorUtil;

            if (debug) std::cout << "Loading event\n";

            TLorentzVector t_0_p4, t_1_p4, bjet0_p4, bjet1_p4, met_p4, svFit_p4;
            t_0_p4.SetPxPyPzE(eventbase.GetLeg(2).GetMomentum().Px(), eventbase.GetLeg(2).GetMomentum().Py(), eventbase.GetLeg(2).GetMomentum().Pz(), eventbase.GetLeg(2).GetMomentum().E()); //Todo: Check ordering
            t_1_p4.SetPxPyPzE(eventbase.GetLeg(1).GetMomentum().Px(), eventbase.GetLeg(1).GetMomentum().Py(), eventbase.GetLeg(1).GetMomentum().Pz(), eventbase.GetLeg(1).GetMomentum().E());
            bjet0_p4.SetPxPyPzE(eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().Px(), eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().Py(), eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().Pz(), eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().E());
            bjet1_p4.SetPxPyPzE(eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().Px(), eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().Py(), eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().Pz(), eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().E());
            met_p4.SetPxPyPzE(eventbase.GetMET().GetMomentum().Px(), eventbase.GetMET().GetMomentum().Py(), eventbase.GetMET().GetMomentum().Pz(), eventbase.GetMET().GetMomentum().E());
            svFit_p4.SetPxPyPzE(eventbase.GetHiggsTTMomentum(true).Px(), eventbase.GetHiggsTTMomentum(true).Py(), eventbase.GetHiggsTTMomentum(true).Pz(), eventbase.GetHiggsTTMomentum(true).E());

            //b-jet info
            features["b_0_csv"] = static_cast<double>(eventbase.GetHiggsBB().GetFirstDaughter()->csv());
            features["b_0_rawf"] = static_cast<double>(eventbase.GetHiggsBB().GetFirstDaughter()->rawf());
            features["b_0_mva"] = static_cast<double>(eventbase.GetHiggsBB().GetFirstDaughter()->mva());

            features["b_1_csv"] = static_cast<double>(eventbase.GetHiggsBB().GetSecondDaughter()->csv());
            features["b_1_rawf"] = static_cast<double>(eventbase.GetHiggsBB().GetSecondDaughter()->rawf());
            features["b_1_mva"] = static_cast<double>(eventbase.GetHiggsBB().GetSecondDaughter()->mva());

            //Order jets by pT
            if (bjet0_p4.Pt() < bjet1_p4.Pt()) {
                bjet0_p4.SetPxPyPzE(eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().Px(), eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().Py(), eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().Pz(), eventbase.GetHiggsBB().GetSecondDaughter().GetMomentum().E());
                features["b_0_csv"] = static_cast<double>(eventbase.GetHiggsBB().GetSecondDaughter()->csv());
                features["b_0_rawf"] = static_cast<double>(eventbase.GetHiggsBB().GetSecondDaughter()->rawf());
                features["b_0_mva"] = static_cast<double>(eventbase.GetHiggsBB().GetSecondDaughter()->mva());

                bjet1_p4.SetPxPyPzE(eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().Px(), eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().Py(), eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().Pz(), eventbase.GetHiggsBB().GetFirstDaughter().GetMomentum().E());
                features["b_1_csv"] = static_cast<double>(eventbase.GetHiggsBB().GetFirstDaughter()->csv());
                features["b_1_rawf"] = static_cast<double>(eventbase.GetHiggsBB().GetFirstDaughter()->rawf());
                features["b_1_mva"] = static_cast<double>(eventbase.GetHiggsBB().GetFirstDaughter()->mva());
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
            features["nJets"] = static_cast<double>(eventbase.GetNJets());
            double hT_jets = 0;
            for (const JetCandidate& jet : eventbase.GetJets()) {
                hT_jets += jet.GetMomentum().Et();
            }
            features["hT_jets"] = hT_jets;

            //MET
            features["met_px"] = met_p4.Px();
            features["met_py"] = met_p4.Py();
            features["met_pT"] = met_p4.Pt();

            //Taus
            features["t_0_px"] = t_0_p4.Px();
            features["t_0_py"] = t_0_p4.Py();
            features["t_0_pz"] = t_0_p4.Pz();
            features["t_0_P"] = t_0_p4.P();
            features["t_0_E"] = t_0_p4.E();
            features["t_0_mass"] = t_0_p4.M();
            features["t_0_mT"] = Calculate_MT(t_0_p4, met_p4);

            features["t_1_px"] = t_1_p4.Px();
            features["t_1_py"] = t_1_p4.Py();
            features["t_1_pz"] = t_1_p4.Pz();
            features["t_1_P"] = t_1_p4.P();
            features["t_1_E"] = t_1_p4.E();
            features["t_1_mass"] = t_1_p4.M();
            features["t_1_mT"] = Calculate_MT(t_1_p4, met_p4);

            //Jets
            features["b_0_px"] = bjet0_p4.Px();
            features["b_0_py"] = bjet0_p4.Py();
            features["b_0_pz"] = bjet0_p4.Pz();
            features["b_0_P"] = bjet0_p4.P();
            features["b_0_E"] = bjet0_p4.E();
            features["b_0_mass"] = bjet0_p4.M();

            features["b_1_px"] = bjet1_p4.Px();
            features["b_1_py"] = bjet1_p4.Py();
            features["b_1_pz"] = bjet1_p4.Pz();
            features["b_1_P"] = bjet1_p4.P();
            features["b_1_E"] = bjet1_p4.E();
            features["b_1_mass"] = bjet1_p4.M();

            //SVFit
            features["h_tt_svFit_px"] = svFit_p4.Px();
            features["h_tt_svFit_py"] = svFit_p4.Py();
            features["h_tt_svFit_pz"] = svFit_p4.Pz();
            features["h_tt_svFit_P"] = svFit_p4.P();
            features["h_tt_svFit_E"] = svFit_p4.E();
            features["h_tt_svFit_mass"] = svFit_p4.M();
            features["h_tt_svFit_mT"] = Calculate_MT(eventbase.GetHiggsTTMomentum(true), eventbase.GetMET().GetMomentum());

            //KinFit
            features["diH_kinFit_mass"] = static_cast<double>(eventbase.GetKinFitResults().mass);
            features["diH_kinFit_chi2"] = static_cast<double>(eventbase.GetKinFitResults().chi2);
            features["diH_kinFit_conv"] = static_cast<double>(eventbase.GetKinFitResults().convergence);

            //h->bb
            features["h_bb_px"] = hbb_p4.Px();
            features["h_bb_py"] = hbb_p4.Py();
            features["h_bb_pz"] = hbb_p4.Pz();
            features["h_bb_P"] = hbb_p4.P();
            features["h_bb_E"] = hbb_p4.E();
            features["h_bb_mass"] = hbb_p4.M();

            //h->tautau
            features["h_tt_px"] = htt_p4.Px();
            features["h_tt_py"] = htt_p4.Py();
            features["h_tt_pz"] = htt_p4.Pz();
            features["h_tt_P"] = htt_p4.P();
            features["h_tt_E"] = htt_p4.E();
            features["h_tt_mass"] = htt_p4.M();

            //Di-higgs
            features["diH_px"] = hh_p4.Px();
            features["diH_py"] = hh_p4.Py();
            features["diH_pz"] = hh_p4.Pz();
            features["diH_P"] = hh_p4.P();
            features["diH_E"] = hh_p4.E();
            features["diH_mass"] = hh_p4.M();

            //Shapes
            double hT, sT, centrality, eVis;
            getGlobalEventInfo(&t_0_p4, &t_1_p4, &bjet0_p4, &bjet0_p4, &met_p4,
                &hT, &sT, &centrality, &eVis);
            features["hT"] = hT;
            features["sT"] = sT;
            features["centrality"] = centrality;
            features["eVis"] = eVis;

            /*double sphericity, spherocity, aplanarity, aplanority, upsilon, dShape,
                sphericityEigen0, sphericityEigen1, sphericityEigen2,
                spherocityEigen0, spherocityEigen1, spherocityEigen2;
            getPrimaryEventShapes(&t_0_p4, &t_1_p4, &bjet0_p4, &bjet0_p4,
                &sphericity, &spherocity,
                &aplanarity, &aplanority,
                &upsilon, &dShape,
                &sphericityEigen0, &sphericityEigen1, &sphericityEigen2,
                &spherocityEigen0, &spherocityEigen1, &spherocityEigen2);
            features["sphericity"] = sphericity;
            features["spherocity"] = spherocity;
            features["aplanarity"] = aplanarity;
            features["aplanority"] = aplanority;
            features["upsilon"] = upsilon;
            features["dShape"] = dShape;
            features["sphericityEigen0"] = sphericityEigen0;
            features["sphericityEigen1"] = sphericityEigen1;
            features["sphericityEigen2"] = sphericityEigen2;
            features["spherocityEigen0"] = spherocityEigen0;
            features["spherocityEigen1"] = spherocityEigen1;
            features["spherocityEigen2"] = spherocityEigen2;

            //Twist
            features["twist_b_0_b_1"] = atan(std::abs(DeltaPhi(bjet0_p4, bjet1_p4)/(bjet0_p4.Eta()-bjet1_p4.Eta())));
            features["twist_b_0_t_0"] = atan(std::abs(DeltaPhi(bjet0_p4, t_0_p4)/(bjet0_p4.Eta()-t_0_p4.Eta())));
            features["twist_b_0_t_1"] = atan(std::abs(DeltaPhi(bjet0_p4, t_1_p4)/(bjet0_p4.Eta()-t_1_p4.Eta())));
            features["twist_b_1_t_0"] = atan(std::abs(DeltaPhi(bjet1_p4, t_0_p4)/(bjet1_p4.Eta()-t_0_p4.Eta())));
            features["twist_b_1_t_1"] = atan(std::abs(DeltaPhi(bjet1_p4, t_1_p4)/(bjet1_p4.Eta()-t_1_p4.Eta())));
            features["twist_t_0_t_1"] = atan(std::abs(DeltaPhi(t_0_p4, t_1_p4)/(t_0_p4.Eta()-t_1_p4.Eta())));
            features["twist_h_bb_h_tt"] = atan(std::abs(DeltaPhi(hbb_p4, htt_p4)/(hbb_p4.Eta()-htt_p4.Eta())));

            //dR
            features["dR_b_0_b_1"] = DeltaR(bjet0_p4, bjet1_p4);
            features["dR_b_0_t_0"] = DeltaR(bjet0_p4, t_0_p4);
            features["dR_b_0_t_1"] = DeltaR(bjet0_p4, t_1_p4);
            features["dR_b_1_t_0"] = DeltaR(bjet1_p4, t_0_p4);
            features["dR_b_1_t_1"] = DeltaR(bjet1_p4, t_1_p4);
            features["dR_t_0_t_1"] = DeltaR(t_0_p4, t_1_p4);
            features["dR_h_bb_h_tt"] = DeltaR(hbb_p4, htt_p4);*/

            if (debug) std::cout << "Event loaded, populating input tensor\n";
            for (size_t i = 0; i < inputFeatures.size(); i++) { //Load selected input features into tensor with standardisation and nromalisation
                if (debug) std::cout << "Feature " << i << ": " << inputFeatures[i] << " = (" << features[inputFeatures[i]] << " - " << means[i] << ")/" << scales[i] << " = " << (features[inputFeatures[i]] - means[i])/scales[i] << " as float: " << static_cast<float>((features[inputFeatures[i]] - means[i])/scales[i])<< "\n";
                input.matrix<float>()(0, static_cast<Eigen::Index>(i)) = static_cast<float>((features[inputFeatures[i]] - means[i])/scales[i]);
            }
            if (debug) std::cout << "Input tensor populated\n";
        }

        double Evaluate() override {
            if (debug) {
                int node_count = graphDef->node_size();
                for (int i = 0; i < node_count; i++) {
                        auto n = graphDef->node(i);
                        std::cout<<"Names : "<< n.name() <<std::endl;

                }
            }

            std::vector<tensorflow::Tensor> outputs;
            if (debug) std::cout << "Evaluating event\n";
            tensorflow::run(session, { { "inputs", input } }, { "output_node0" }, &outputs);
            if (debug) std::cout << "Event evaulated, class prediction is: " << outputs[0].matrix<float>()(0, 0) << "\n";
            return outputs[0].matrix<float>()(0, 0);
        }

        std::shared_ptr<TMVA::Reader> GetReader() override {throw exception ("GetReader not supported.");}
};

} //mva_study
} //analysis