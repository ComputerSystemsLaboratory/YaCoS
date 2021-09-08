/*
Copyright 2021 Anderson Faustino da Silva

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <iostream>
#include <sstream>

#include "llvm/IR/InstIterator.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "llvm_loop_funcinfo.h"

using namespace ::llvm;

namespace compy {
namespace llvm {
namespace loop {

void FunctionInfoPass::initFeatures(){
    if (!features_.empty())
      features_.clear();
    this->updateFeature("ft01_loopWithWR", 0.0);
    this->updateFeature("ft02_loopWRarray", 0.0);
    this->updateFeature("ft03_loopWRarrayDisj", 0.0);
    this->updateFeature("ft04_isFlowDep", 0.0);
    this->updateFeature("ft05_isAntiDep", 0.0);
    this->updateFeature("ft06_isInputDep", 0.0);
    this->updateFeature("ft07_isOutputDep", 0.0);
    this->updateFeature("ft08_simpleLoop", 0.0);
    this->updateFeature("ft09_complexLoop", 0.0);
    this->updateFeature("ft10_loopRAW", 0.0);
    this->updateFeature("ft11_loopWAR", 0.0);
    this->updateFeature("ft12_loopWAW", 0.0);
    this->updateFeature("ft13_loopRAR", 0.0);
    this->updateFeature("ft14_indepLoopDep", 0.0);
    this->updateFeature("ft15_loopSelfDep", 0.0);
    this->updateFeature("ft16_numOfLoops", 0.0);

}

void FunctionInfoPass::updateFeature(std::string  key, float value){
    std::map<std::string, float>::iterator it = features_.find(key);
    if (it == features_.end())
      features_[key] = value;
    else
      it->second += value;
}

void FunctionInfoPass::DependenceCheckFunctionIntern(const Function &F, DependenceInfo &DI,
						    std::vector<Instruction*> loopInsts) {

  unsigned indepLoopTemp = 0, isInputTemp = 0,
    isOutputTemp = 0, isFlowTemp = 0,
    isAntiTemp = 0, loopSelfTemp = 0;
  unsigned vecSize = loopInsts.size();

  for (unsigned int I = 0; I < vecSize; I++) {
    for (unsigned int J = I; J != vecSize; J++) {
      std::unique_ptr<Dependence> infoPtr;
      infoPtr = DI.depends(loopInsts[I], loopInsts[J], true);
      Dependence *dep = infoPtr.get();

      if (dep != NULL) {
	Instruction* srcInt = dep->getSrc();
	Instruction* dstInt = dep->getDst();

	if (srcInt == dstInt) {
	  //dep->getDst()->print(errs(), false);
	  //errs() << "   ---> ";
	  //dep->getSrc()->print(errs(), false);
	  //errs() << "\n";
	  ++loopSelfTemp;
	}
      }
      // isInput(RAR): read/read
      if (dep != NULL && dep->isInput()) {
	if (dep->isLoopIndependent()) {
	  ++indepLoopTemp ;
	}
	++isInputTemp;
      }
      // isOutput(WAW): write/write
      if (dep != NULL && dep->isOutput()) {
	if (dep->isLoopIndependent()) {
	  ++indepLoopTemp;
	}
	++isOutputTemp;
      }
      // isFlow(RAW): write/read
      if (dep != NULL && dep->isFlow()) {
	if (dep->isLoopIndependent()) {
	  ++indepLoopTemp;
	}
	++isFlowTemp;
      }
      // isAnti(WAR): read/write
      if (dep != NULL && dep->isAnti()) {
	if (dep->isLoopIndependent()) {
	  ++indepLoopTemp;
	}
	++isAntiTemp;
      }
    }
  }
  if (loopSelfTemp != 0) this->updateFeature("ft15_loopSelfDep", 1.0);
  if (indepLoopTemp != 0)  this->updateFeature("ft14_indepLoopDep", 1.0);
  if (isInputTemp != 0)  this->updateFeature("ft13_loopRAR", 1.0);
  if (isOutputTemp != 0)  this->updateFeature("ft12_loopWAW", 1.0);
  if (isFlowTemp != 0)  this->updateFeature("ft10_loopRAW", 1.0);
  if (isAntiTemp != 0)  this->updateFeature("ft11_loopWAR", 1.0);
}

/* Esta função só é chamada no loop mais exterior, não havendo
     a necessidade de contar novamente para loops interos, pois
     toda a informação de loops internos já é presente no loop
     exterior.
  */
void FunctionInfoPass::DependenceCheckFunctionExtern(const Function &F, DependenceInfo &DI,
						  std::vector<Instruction*> loopInsts) {

  unsigned vecSize = loopInsts.size();
  for (unsigned int I = 0; I < vecSize; I++) {
    for (unsigned int J = I; J != vecSize; J++) {
      std::unique_ptr<Dependence> infoPtr;
      infoPtr = DI.depends(loopInsts[I], loopInsts[J], true);
      Dependence *dep = infoPtr.get();
      if (dep != NULL && dep->isInput()) {
	       this->updateFeature("ft06_isInputDep", 1.0);
      }
      if (dep != NULL && dep->isOutput()) {
	       this->updateFeature("ft07_isOutputDep", 1.0);
      }
      if (dep != NULL && dep->isFlow()) {
	       this->updateFeature("ft04_isFlowDep", 1.0);
      }
      if (dep != NULL && dep->isAnti()) {
	       this->updateFeature("ft05_isAntiDep", 1.0);
      }
    }
  }
}

/* Loops with disjoint set of array reads/write:
   - Deve ser analisado no MESMO loop, se es possui
   r/w a arrays disjuntos
   Loops with array read/writes to same array:
   -
*/
void FunctionInfoPass::RecursiveIterLoopFramework(const Function &F, DependenceInfo &DI,
           Loop *L, std::vector<Instruction*> defUseOfArrays, unsigned nesting) {
  unsigned loopWithWRLocal = 0, loopWRarrayLocal = 0;
  std::set<Instruction*> disjointStore;
  std::set<Instruction*> disjointLoad;
  std::vector<Instruction*> loopInsts;

  this->updateFeature("ft16_numOfLoops", 1.0);

  Loop::block_iterator bb;

  unsigned arrSize = defUseOfArrays.size();
  /* Iterando sob os blocos básicos e instruções
   de cada loop. A análise se inicia pelo loop mais
   exterior e procede para Loops Internos. Um loop
   exterior pode possuir ou não loop alinhados.
   */
   for (bb = L->block_begin(); bb != L->block_end();++bb) {
     BasicBlock* block = *bb;
     for (BasicBlock::iterator BI = block->begin(),
      BE = block->end(); BI != BE; BI++) {
        loopInsts.push_back(&*BI);

        if(auto* LI = dyn_cast<LoadInst>(BI)) {
          ++loopWithWRLocal;
        }
        else if(auto* LS = dyn_cast<StoreInst>(BI)) {
          ++loopWithWRLocal;
        }

        /* O uso desta técnica com o vetor defUseOfArrays é
        equivalentes a verificar as instruções dentro
        dos blocos básicos, pois defUseOfArrays já contem
        todas as instruções Load/Store (contendo instruções)
        que não são contempladas em Loops, ou seja, defUseOfArrays
        já possui tudo que seria posteriormente analisado aqui.
        */
        if (auto* SI = dyn_cast<StoreInst>(BI)) {
          for (unsigned int i = 0; i<arrSize; i++) {
            if (defUseOfArrays[i]==SI) {
              disjointStore.insert(SI);
              ++loopWRarrayLocal;
            }
          }
        }

        if (auto* LI = dyn_cast<LoadInst>(BI)) {
          for (unsigned int i = 0; i<arrSize; i++) {
            if (defUseOfArrays[i]==LI) {
              disjointLoad.insert(LI);
              ++loopWRarrayLocal;
            }
          }
        }
      }
   }

   /* Não contempla 1 load ^ 1 store (??) */
   if ((disjointStore.size()>1) || (disjointLoad.size()>1)) {
     this->updateFeature("ft03_loopWRarrayDisj", 1.0);
   }

   disjointLoad.clear();
   disjointStore.clear();

   if (loopWithWRLocal != 0) {
     this->updateFeature("ft01_loopWithWR", 1.0);
   }
   if (loopWRarrayLocal != 0) {
     this->updateFeature("ft02_loopWRarray", 1.0);
   }

   /* loopInsts salva todas as instruções do loop analisado
   no momento para este ser utilizado na análise de dependência.
   Como a função RecursiveIterLoopFramework, o vetor é limpo
   apos o retorno da função DependenceCheckFunctionIntern
   */
   DependenceCheckFunctionIntern(F, DI, loopInsts);
   loopInsts.clear();
   std::vector<Loop*> subLoops = L->getSubLoops();
   Loop::iterator j, f;
   for (j = subLoops.begin(), f = subLoops.end(); j != f; ++j) {
     RecursiveIterLoopFramework(F, DI, *j, defUseOfArrays,nesting + 1);
   }

   /* Armazenando níveis */
   tracking_.push_back(nesting);
}

void FunctionInfoPass::FeaturesExtractor(Function &F, DependenceInfo &DI, LoopInfo &LI) {

  /* loopInsts: Vetor que possui apenas as intruções do Loop
    functionInsts: Vetor que possui todas as instruções da função
    defUseOfArrays: Vetor que possui apenas Store/Loads de Arrays
    diffInsts: Vetor que possui instruções que não pertencem a
    um Loop mais exterior (analisado no for(LoopInfo..))
  */

  std::vector<int> numOfLevels;
  std::vector<Instruction*> loopInsts;
  std::vector<Instruction*> functionInsts;
  std::vector<Instruction*> defUseOfArrays;
  std::vector<Instruction*> diffInsts;

  /* Este método funciona da seguinte forma:
      1) Faço uma varredura das instruções;
      2) Econtro instruções GEP;
                2.1) Se GEP é um Array: Salvo
                2.2) Se não, descarto;
      3) Faço uma cadeia def/use com o Gep salvo;
      4) Salvo em um vetor, todas os usos (use) que são
     Stores/Loads (write/read) de Gep salvo (def)
     ------------------------------------------------
    No algoritmo recursivo de Loop, se as Instruções,
    pertencentes ao loop analisado forem uma das salvas
    no vetor, então, aquele é um loop com R/W em Arrays.
  */

  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; I++) {
    if(auto *GEP = dyn_cast<GetElementPtrInst>(&*I)) {
      Type *T = cast<PointerType>(cast<GetElementPtrInst>(GEP)->getPointerOperandType())->getElementType();
      if(isa<ArrayType>(T)) {
        Instruction* instt = &*I;
        for (User *U : instt->users()) {
            if (Instruction *Inst = dyn_cast<Instruction>(U)) {
              if (auto* LIC = dyn_cast<LoadInst>(Inst)) {
                defUseOfArrays.push_back(LIC);
              }
              if (auto* SIC = dyn_cast<StoreInst>(Inst)) {
                defUseOfArrays.push_back(SIC);
              }
            }
        }
      }
    }
    functionInsts.push_back(&*I);
  }

  for (LoopInfo::iterator i = LI.begin(), e = LI.end(); i != e; ++i) {
    Loop *L = *i;
    Loop::block_iterator bb;

    /* Computando loopInsts */
    for (bb = L->block_begin(); bb != L->block_end();++bb) {
      BasicBlock* block = *bb;
      for (BasicBlock::iterator BI = block->begin(), BE = block->end(); BI != BE; BI++) {
        loopInsts.push_back(&*BI);
      }
    }

    /* Computando diffInsts */
    std::set_difference(functionInsts.begin(), functionInsts.end(),
    loopInsts.begin(), loopInsts.end(), std::inserter(diffInsts, diffInsts.begin()));

    /* Computando Dependencias (Intern/Extern) */
    DependenceCheckFunctionExtern(F, DI, loopInsts);
    RecursiveIterLoopFramework(F, DI, *i, defUseOfArrays,0);

    /* SimpleLoop e ComplexLoop só são unicamente para o loop EXTERIOR !*/
    std::sort(std::begin(tracking_), std::end(tracking_));
    auto pos = std::adjacent_find(std::begin(tracking_), std::end(tracking_));
    if ( pos != std::end(tracking_)) {
      this->updateFeature("ft09_complexLoop", 1.0);
    }
    else {
      this->updateFeature("ft08_simpleLoop", 1.0);
    }

    /* Limpando o nível e instruções do Loop Exterior */
    tracking_.clear();
    loopInsts.clear();
  }

  defUseOfArrays.clear();

}

bool FunctionInfoPass::runOnFunction(::llvm::Function &func) {
  // create a new info object and invalidate the old one
  info_ = FunctionInfoPtr(new FunctionInfo());
  info_->name = func.getName().str();

  this->initFeatures();

  // Extract features
  //DependenceInfo &DI = getAnalysis<DependenceAnalysisWrapperPass>().getDI();
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  //FeaturesExtractor(func, DI, LI);
  // End Extract Features

  info_->features = features_;

  // indicate that nothing was changed
  return false;
}

void FunctionInfoPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<LoopInfoWrapperPass>();
  au.addRequired<TargetTransformInfoWrapperPass>();
  //au.addRequired<DependenceAnalysisWrapperPass>();
  au.setPreservesAll();
}

char FunctionInfoPass::ID = 0;

static RegisterPass<FunctionInfoPass> X("loopfuncinfo", "Function Loop Features Extractor",
                                        true /* Only looks at CFG */,
                                        true /* Analysis Pass */);
/*static RegisterStandardPasses
    RegisterFunctionInfoPass(PassManagerBuilder::EP_EarlyAsPossible,
                          [](const PassManagerBuilder &Builder,
                             legacy::PassManagerBase &PM) {
                            PM.add(new FunctionInfoPass());
                          });*/

}  // namespace loop
}  // namespace llvm
}  // namespace compy
