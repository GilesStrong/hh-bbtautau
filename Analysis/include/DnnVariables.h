/*! Definition of DnnMvaVariables.
This file is part of https://github.com/hh-italian-group/hh-bbtautau. */

#pragma once

#include "MvaVariables.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "TMatrixD.h"
#include "TMatrixT.h"
#include "TMatrixDEigen.h"

#ifndef ROOT_TMatrixT
 #define ROOT_TMatrixT
 
 //////////////////////////////////////////////////////////////////////////
 //                                                                      //
 // TMatrixT                                                             //
 //                                                                      //
 // Template class of a general matrix in the linear algebra package     //
 //                                                                      //
 //////////////////////////////////////////////////////////////////////////
 
 #ifndef ROOT_TMatrixTBase
 #include "TMatrixTBase.h"
 #endif
 #ifndef ROOT_TMatrixTUtils
 #include "TMatrixTUtils.h"
 #endif
 
 #ifdef CBLAS
 #include <vecLib/vBLAS.h>
 //#include <cblas.h>
 #endif
 
 
 template<class Element> class TMatrixTSym;
 template<class Element> class TMatrixTSparse;
 template<class Element> class TMatrixTLazy;
 
 template<class Element> class TMatrixT : public TMatrixTBase<Element> {
 
 protected:
 
    Element  fDataStack[TMatrixTBase<Element>::kSizeMax]; //! data container
    Element *fElements;                                   //[fNelems] elements themselves
 
    Element *New_m   (Int_t size);
    void     Delete_m(Int_t size,Element*&);
    Int_t    Memcpy_m(Element *newp,const Element *oldp,Int_t copySize,
                       Int_t newSize,Int_t oldSize);
    void     Allocate(Int_t nrows,Int_t ncols,Int_t row_lwb = 0,Int_t col_lwb = 0,Int_t init = 0,
                      Int_t /*nr_nonzeros*/ = -1);
 
 
 public:
 
 
    enum {kWorkMax = 100};
    enum EMatrixCreatorsOp1 { kZero,kUnit,kTransposed,kInverted,kAtA };
    enum EMatrixCreatorsOp2 { kMult,kTransposeMult,kInvMult,kMultTranspose,kPlus,kMinus };
 
    TMatrixT(): fDataStack(), fElements(0) { }
    TMatrixT(Int_t nrows,Int_t ncols);
    TMatrixT(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb);
    TMatrixT(Int_t nrows,Int_t ncols,const Element *data,Option_t *option="");
    TMatrixT(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,const Element *data,Option_t *option="");
    TMatrixT(const TMatrixT      <Element> &another);
    TMatrixT(const TMatrixTSym   <Element> &another);
    TMatrixT(const TMatrixTSparse<Element> &another);
    template <class Element2> TMatrixT(const TMatrixT<Element2> &another): fElements(0)
    {
       R__ASSERT(another.IsValid());
       Allocate(another.GetNrows(),another.GetNcols(),another.GetRowLwb(),another.GetColLwb());
       *this = another;
    }
 
    TMatrixT(EMatrixCreatorsOp1 op,const TMatrixT<Element> &prototype);
    TMatrixT(const TMatrixT    <Element> &a,EMatrixCreatorsOp2 op,const TMatrixT   <Element> &b);
    TMatrixT(const TMatrixT    <Element> &a,EMatrixCreatorsOp2 op,const TMatrixTSym<Element> &b);
    TMatrixT(const TMatrixTSym <Element> &a,EMatrixCreatorsOp2 op,const TMatrixT   <Element> &b);
    TMatrixT(const TMatrixTSym <Element> &a,EMatrixCreatorsOp2 op,const TMatrixTSym<Element> &b);
    TMatrixT(const TMatrixTLazy<Element> &lazy_constructor);
 
    virtual ~TMatrixT() { Clear(); }
 
    // Elementary constructors
 
    void Plus (const TMatrixT   <Element> &a,const TMatrixT   <Element> &b);
    void Plus (const TMatrixT   <Element> &a,const TMatrixTSym<Element> &b);
    void Plus (const TMatrixTSym<Element> &a,const TMatrixT   <Element> &b) { Plus(b,a); }
 
    void Minus(const TMatrixT   <Element> &a,const TMatrixT   <Element> &b);
    void Minus(const TMatrixT   <Element> &a,const TMatrixTSym<Element> &b);
    void Minus(const TMatrixTSym<Element> &a,const TMatrixT   <Element> &b) { Minus(b,a); }
 
    void Mult (const TMatrixT   <Element> &a,const TMatrixT   <Element> &b);
    void Mult (const TMatrixT   <Element> &a,const TMatrixTSym<Element> &b);
    void Mult (const TMatrixTSym<Element> &a,const TMatrixT   <Element> &b);
    void Mult (const TMatrixTSym<Element> &a,const TMatrixTSym<Element> &b);
 
    void TMult(const TMatrixT   <Element> &a,const TMatrixT   <Element> &b);
    void TMult(const TMatrixT   <Element> &a,const TMatrixTSym<Element> &b);
    void TMult(const TMatrixTSym<Element> &a,const TMatrixT   <Element> &b) { Mult(a,b); }
    void TMult(const TMatrixTSym<Element> &a,const TMatrixTSym<Element> &b) { Mult(a,b); }
 
    void MultT(const TMatrixT   <Element> &a,const TMatrixT   <Element> &b);
    void MultT(const TMatrixT   <Element> &a,const TMatrixTSym<Element> &b) { Mult(a,b); }
    void MultT(const TMatrixTSym<Element> &a,const TMatrixT   <Element> &b);
    void MultT(const TMatrixTSym<Element> &a,const TMatrixTSym<Element> &b) { Mult(a,b); }
 
    virtual const Element *GetMatrixArray  () const;
    virtual       Element *GetMatrixArray  ();
    virtual const Int_t   *GetRowIndexArray() const { return 0; }
    virtual       Int_t   *GetRowIndexArray()       { return 0; }
    virtual const Int_t   *GetColIndexArray() const { return 0; }
    virtual       Int_t   *GetColIndexArray()       { return 0; }
 
    virtual       TMatrixTBase<Element> &SetRowIndexArray(Int_t * /*data*/) { MayNotUse("SetRowIndexArray(Int_t *)"); return *this; }
    virtual       TMatrixTBase<Element> &SetColIndexArray(Int_t * /*data*/) { MayNotUse("SetColIndexArray(Int_t *)"); return *this; }
 
    virtual void Clear(Option_t * /*option*/ ="") { if (this->fIsOwner) Delete_m(this->fNelems,fElements);
                                                    else fElements = 0;
                                                    this->fNelems = 0; }
 
            TMatrixT    <Element> &Use     (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Element *data);
    const   TMatrixT    <Element> &Use     (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,const Element *data) const
                                             { return (const TMatrixT<Element>&)
                                                      ((const_cast<TMatrixT<Element> *>(this))->Use(row_lwb,row_upb,col_lwb,col_upb, const_cast<Element *>(data))); }
            TMatrixT    <Element> &Use     (Int_t nrows,Int_t ncols,Element *data);
    const   TMatrixT    <Element> &Use     (Int_t nrows,Int_t ncols,const Element *data) const;
            TMatrixT    <Element> &Use     (TMatrixT<Element> &a);
    const   TMatrixT    <Element> &Use     (const TMatrixT<Element> &a) const;
 
    virtual TMatrixTBase<Element> &GetSub  (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                            TMatrixTBase<Element> &target,Option_t *option="S") const;
            TMatrixT    <Element>  GetSub  (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Option_t *option="S") const;
    virtual TMatrixTBase<Element> &SetSub  (Int_t row_lwb,Int_t col_lwb,const TMatrixTBase<Element> &source);
 
    virtual TMatrixTBase<Element> &ResizeTo(Int_t nrows,Int_t ncols,Int_t /*nr_nonzeros*/ =-1);
    virtual TMatrixTBase<Element> &ResizeTo(Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,Int_t /*nr_nonzeros*/ =-1);
    inline  TMatrixTBase<Element> &ResizeTo(const TMatrixT<Element> &m) {
                                             return ResizeTo(m.GetRowLwb(),m.GetRowUpb(),m.GetColLwb(),m.GetColUpb());
                                  }
 
    virtual Double_t Determinant  () const;
    virtual void     Determinant  (Double_t &d1,Double_t &d2) const;
 
            TMatrixT<Element> &Invert      (Double_t *det=0);
            TMatrixT<Element> &InvertFast  (Double_t *det=0);
            TMatrixT<Element> &Transpose   (const TMatrixT<Element> &source);
    inline  TMatrixT<Element> &T           () { return this->Transpose(*this); }
            TMatrixT<Element> &Rank1Update (const TVectorT<Element> &v,Element alpha=1.0);
            TMatrixT<Element> &Rank1Update (const TVectorT<Element> &v1,const TVectorT<Element> &v2,Element alpha=1.0);
            Element            Similarity  (const TVectorT<Element> &v) const;
 
    TMatrixT<Element> &NormByColumn(const TVectorT<Element> &v,Option_t *option="D");
    TMatrixT<Element> &NormByRow   (const TVectorT<Element> &v,Option_t *option="D");
 
    // Either access a_ij as a(i,j)
    inline       Element                     operator()(Int_t rown,Int_t coln) const;
    inline       Element                    &operator()(Int_t rown,Int_t coln);
 
    // or as a[i][j]
    inline const TMatrixTRow_const<Element>  operator[](Int_t rown) const { return TMatrixTRow_const<Element>(*this,rown); }
    inline       TMatrixTRow      <Element>  operator[](Int_t rown)       { return TMatrixTRow      <Element>(*this,rown); }
 
    TMatrixT<Element> &operator= (const TMatrixT      <Element> &source);
    TMatrixT<Element> &operator= (const TMatrixTSym   <Element> &source);
    TMatrixT<Element> &operator= (const TMatrixTSparse<Element> &source);
    TMatrixT<Element> &operator= (const TMatrixTLazy  <Element> &source);
    template <class Element2> TMatrixT<Element> &operator= (const TMatrixT<Element2> &source)
    {
       if (!AreCompatible(*this,source)) {
          Error("operator=(const TMatrixT2 &)","matrices not compatible");
          return *this;
       }
 
      TObject::operator=(source);
      const Element2 * const ps = source.GetMatrixArray();
            Element  * const pt = this->GetMatrixArray();
      for (Int_t i = 0; i < this->fNelems; i++)
         pt[i] = ps[i];
      this->fTol = source.GetTol();
      return *this;
    }
 
    TMatrixT<Element> &operator= (Element val);
    TMatrixT<Element> &operator-=(Element val);
    TMatrixT<Element> &operator+=(Element val);
    TMatrixT<Element> &operator*=(Element val);
 
    TMatrixT<Element> &operator+=(const TMatrixT   <Element> &source);
    TMatrixT<Element> &operator+=(const TMatrixTSym<Element> &source);
    TMatrixT<Element> &operator-=(const TMatrixT   <Element> &source);
    TMatrixT<Element> &operator-=(const TMatrixTSym<Element> &source);
 
    TMatrixT<Element> &operator*=(const TMatrixT            <Element> &source);
    TMatrixT<Element> &operator*=(const TMatrixTSym         <Element> &source);
    TMatrixT<Element> &operator*=(const TMatrixTDiag_const  <Element> &diag);
    TMatrixT<Element> &operator/=(const TMatrixTDiag_const  <Element> &diag);
    TMatrixT<Element> &operator*=(const TMatrixTRow_const   <Element> &row);
    TMatrixT<Element> &operator/=(const TMatrixTRow_const   <Element> &row);
    TMatrixT<Element> &operator*=(const TMatrixTColumn_const<Element> &col);
    TMatrixT<Element> &operator/=(const TMatrixTColumn_const<Element> &col);
 
    const TMatrixT<Element> EigenVectors(TVectorT<Element> &eigenValues) const;
 
    ClassDef(TMatrixT,4) // Template of General Matrix class
 };
 
 #ifndef __CINT__
 // When building with -fmodules, it instantiates all pending instantiations,
 // instead of delaying them until the end of the translation unit.
 // We 'got away with' probably because the use and the definition of the
 // explicit specialization do not occur in the same TU.
 //
 // In case we are building with -fmodules, we need to forward declare the
 // specialization in order to compile the dictionary G__Matrix.cxx.
 template <> TClass *TMatrixT<double>::Class();
 #endif // __CINT__
 
 
 template <class Element> inline const Element           *TMatrixT<Element>::GetMatrixArray() const { return fElements; }
 template <class Element> inline       Element           *TMatrixT<Element>::GetMatrixArray()       { return fElements; }
 
 template <class Element> inline       TMatrixT<Element> &TMatrixT<Element>::Use           (Int_t nrows,Int_t ncols,Element *data)
                                                                                           { return Use(0,nrows-1,0,ncols-1,data); }
 template <class Element> inline const TMatrixT<Element> &TMatrixT<Element>::Use           (Int_t nrows,Int_t ncols,const Element *data) const
                                                                                           { return Use(0,nrows-1,0,ncols-1,data); }
 template <class Element> inline       TMatrixT<Element> &TMatrixT<Element>::Use           (TMatrixT &a)
                                                                                           {
                                                                                             R__ASSERT(a.IsValid());
                                                                                             return Use(a.GetRowLwb(),a.GetRowUpb(),
                                                                                                        a.GetColLwb(),a.GetColUpb(),a.GetMatrixArray());
                                                                                           }
 template <class Element> inline const TMatrixT<Element> &TMatrixT<Element>::Use           (const TMatrixT &a) const
                                                                                           {
                                                                                             R__ASSERT(a.IsValid());
                                                                                             return Use(a.GetRowLwb(),a.GetRowUpb(),
                                                                                                        a.GetColLwb(),a.GetColUpb(),a.GetMatrixArray());
                                                                                           }
 
 template <class Element> inline       TMatrixT<Element>  TMatrixT<Element>::GetSub        (Int_t row_lwb,Int_t row_upb,Int_t col_lwb,Int_t col_upb,
                                                                                            Option_t *option) const
                                                                                           {
                                                                                             TMatrixT tmp;
                                                                                             this->GetSub(row_lwb,row_upb,col_lwb,col_upb,tmp,option);
                                                                                             return tmp;
                                                                                           }
 
 template <class Element> inline Element TMatrixT<Element>::operator()(Int_t rown,Int_t coln) const
 {
    R__ASSERT(this->IsValid());
    const Int_t arown = rown-this->fRowLwb;
    const Int_t acoln = coln-this->fColLwb;
    if (arown >= this->fNrows || arown < 0) {
       Error("operator()","Request row(%d) outside matrix range of %d - %d",rown,this->fRowLwb,this->fRowLwb+this->fNrows);
       return TMatrixTBase<Element>::NaNValue();
    }
    if (acoln >= this->fNcols || acoln < 0) {
       Error("operator()","Request column(%d) outside matrix range of %d - %d",coln,this->fColLwb,this->fColLwb+this->fNcols);
       return TMatrixTBase<Element>::NaNValue();
 
    }
    return (fElements[arown*this->fNcols+acoln]);
 }
 
 template <class Element> inline Element &TMatrixT<Element>::operator()(Int_t rown,Int_t coln)
 {
    R__ASSERT(this->IsValid());
    const Int_t arown = rown-this->fRowLwb;
    const Int_t acoln = coln-this->fColLwb;
    if (arown >= this->fNrows || arown < 0) {
       Error("operator()","Request row(%d) outside matrix range of %d - %d",rown,this->fRowLwb,this->fRowLwb+this->fNrows);
       return TMatrixTBase<Element>::NaNValue();
    }
    if (acoln >= this->fNcols || acoln < 0) {
       Error("operator()","Request column(%d) outside matrix range of %d - %d",coln,this->fColLwb,this->fColLwb+this->fNcols);
       return TMatrixTBase<Element>::NaNValue();
    }
    return (fElements[arown*this->fNcols+acoln]);
 }
 
 template <class Element> TMatrixT<Element>  operator+  (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator+  (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
 template <class Element> TMatrixT<Element>  operator+  (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator+  (const TMatrixT   <Element> &source ,      Element               val    );
 template <class Element> TMatrixT<Element>  operator+  (      Element               val    ,const TMatrixT   <Element> &source );
 template <class Element> TMatrixT<Element>  operator-  (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator-  (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
 template <class Element> TMatrixT<Element>  operator-  (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator-  (const TMatrixT   <Element> &source ,      Element               val    );
 template <class Element> TMatrixT<Element>  operator-  (      Element               val    ,const TMatrixT   <Element> &source );
 template <class Element> TMatrixT<Element>  operator*  (      Element               val    ,const TMatrixT   <Element> &source );
 template <class Element> TMatrixT<Element>  operator*  (const TMatrixT   <Element> &source ,      Element               val    );
 template <class Element> TMatrixT<Element>  operator*  (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator*  (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
 template <class Element> TMatrixT<Element>  operator*  (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator*  (const TMatrixTSym<Element> &source1,const TMatrixTSym<Element> &source2);
 // Preventing warnings with -Weffc++ in GCC since overloading the || and && operators was a design choice.
 #if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
 #pragma GCC diagnostic push
 #pragma GCC diagnostic ignored "-Weffc++"
 #endif
 template <class Element> TMatrixT<Element>  operator&& (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator&& (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
 template <class Element> TMatrixT<Element>  operator&& (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator|| (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator|| (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
 template <class Element> TMatrixT<Element>  operator|| (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
 #if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40600
 #pragma GCC diagnostic pop
 #endif
 template <class Element> TMatrixT<Element>  operator>  (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator>  (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
 template <class Element> TMatrixT<Element>  operator>  (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator>= (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator>= (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
 template <class Element> TMatrixT<Element>  operator>= (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator<= (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator<= (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
 template <class Element> TMatrixT<Element>  operator<= (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator<  (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator<  (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
 template <class Element> TMatrixT<Element>  operator<  (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator!= (const TMatrixT   <Element> &source1,const TMatrixT   <Element> &source2);
 template <class Element> TMatrixT<Element>  operator!= (const TMatrixT   <Element> &source1,const TMatrixTSym<Element> &source2);
 template <class Element> TMatrixT<Element>  operator!= (const TMatrixTSym<Element> &source1,const TMatrixT   <Element> &source2);
 
 template <class Element> TMatrixT<Element> &Add        (TMatrixT<Element> &target,      Element               scalar,const TMatrixT   <Element> &source);
 template <class Element> TMatrixT<Element> &Add        (TMatrixT<Element> &target,      Element               scalar,const TMatrixTSym<Element> &source);
 template <class Element> TMatrixT<Element> &ElementMult(TMatrixT<Element> &target,const TMatrixT   <Element> &source);
 template <class Element> TMatrixT<Element> &ElementMult(TMatrixT<Element> &target,const TMatrixTSym<Element> &source);
 template <class Element> TMatrixT<Element> &ElementDiv (TMatrixT<Element> &target,const TMatrixT   <Element> &source);
 template <class Element> TMatrixT<Element> &ElementDiv (TMatrixT<Element> &target,const TMatrixTSym<Element> &source);
 
 template <class Element> void AMultB (const Element * const ap,Int_t na,Int_t ncolsa,
                                       const Element * const bp,Int_t nb,Int_t ncolsb,Element *cp);
 template <class Element> void AtMultB(const Element * const ap,Int_t ncolsa,
                                       const Element * const bp,Int_t nb,Int_t ncolsb,Element *cp);
 template <class Element> void AMultBt(const Element * const ap,Int_t na,Int_t ncolsa,
                                       const Element * const bp,Int_t nb,Int_t ncolsb,Element *cp);

namespace analysis {
namespace mva_study{

class DnnMvaVariables : public MvaVariablesBase {
    /*Class for evaluating trained DNN stored in Tensorflow protocol buffer (.pb)*/

    private:
        int nInputs; 
        bool fixRotate;
        std::vector<double> means;
        std::vector<double> scales;
        std::map<std::string, double> features;
        std::vector<std::string> inputFeatures;

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
            inputFeatures = std::vector<std::string>{"h_tt_svFit_mass", "t_1_mT", "diH_kinFit_chi2", "b_0_csv", "b_1_csv", "dR_t_0_t_1", "diH_kinFit_mass", "h_bb_mass", "h_bb_px", "hT", "h_tt_mass", "t_0_px", "diH_kinFit_conv", "t_1_px", "dR_b_0_b_1", "t_0_py", "h_tt_svFit_mT", "t_0_mass", "h_tt_svFit_py", "h_tt_svFit_px", "b_1_px", "diH_px", "h_tt_px", "t_0_P", "hT_jets", "met_px", "t_0_mT", "dR_b_0_t_0", "met_pT", "b_1_py", "t_1_E", "diH_mass", "t_0_E", "centrality", "h_bb_py", "h_bb_P", "b_0_mass", "diH_py", "twist_t_0_t_1", "h_tt_py", "b_1_mva", "b_0_mva", "b_0_py", "b_0_px", "dR_h_bb_h_tt", "met_py", "sT", "h_tt_E", "twist_b_0_t_1", "b_1_P", "twist_h_bb_h_tt", "dR_b_1_t_0", "b_1_rawf", "dR_b_0_t_1", "b_0_E", "twist_b_0_b_1", "b_1_pz", "sphericity", "h_tt_svFit_P", "b_0_rawf", "b_1_E", "t_1_mass", "dR_b_1_t_1", "twist_b_0_t_0", "b_1_mass", "aplanarity", "h_bb_E"};

            nInputs = sizeof(inputFeatures); 
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
            const SampleId& /*mass*/ , int /*spin*/, double /*sample_weight = 1.*/, int /*which_test = -1*/) override {
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

            double sphericity, spherocity, aplanarity, aplanority, upsilon, dShape,
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
            features["dR_h_bb_h_tt"] = DeltaR(hbb_p4, htt_p4);

            for (size_t i = 0; i < inputFeatures.size(); i++) { //Load selected input features into tensor with standardisation and nromalisation
                input.matrix<float>()(0, static_cast<Eigen::Index>(i)) = static_cast<float>((features[inputFeatures[i]] - means[i])/scales[i]);
            }
        }

        double Evaluate() override {
            tensorflow::run(session, { { "input", input } }, { "output" }, &outputs);
            return outputs[0].matrix<float>()(0, 0);
        }

        std::shared_ptr<TMVA::Reader> GetReader() override {
            return nullptr;
        }
};

} //mva_study
} //analysis