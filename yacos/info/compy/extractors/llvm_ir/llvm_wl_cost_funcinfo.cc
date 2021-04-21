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

#include "llvm_wl_cost_funcinfo.h"

using namespace ::llvm;

namespace compy {
namespace llvm {
namespace wlcost {

bool FunctionInfoPass::runOnFunction(::llvm::Function &func) {
  // create a new info object and invalidate the old one
  info_ = FunctionInfoPtr(new FunctionInfo());

  info_->name = func.getName().str();
  info_->recipThroughput = 0.0;
  info_->latency = 0.0;
  info_->codeSize = 0.0;
  info_->oneCost = 0.0;

  auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(func);
  BlockEdgeFrequencyPass *BEFP = &getAnalysis<BlockEdgeFrequencyPass>();

  for (auto &bb : func.getBasicBlockList()) {
    double BB_freq = BEFP->getBlockFrequency(&bb);
    for (auto &inst : bb) {
      info_->recipThroughput += BB_freq * TTI->getInstructionCost(
                                                  &inst,
                                                  TTI->TCK_RecipThroughput
                                                );
      info_->latency += BB_freq * TTI->getInstructionCost(
                                                  &inst,
                                                  TTI->TCK_Latency
                                                );
      info_->codeSize += BB_freq * TTI->getInstructionCost(
                                                  &inst,
                                                  TTI->TCK_CodeSize
                                                );
      info_->oneCost += BB_freq * 1.0;
    }
  }

  // indicate that nothing was changed
  return false;
}

void FunctionInfoPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<BlockEdgeFrequencyPass>();
  au.addRequired<TargetTransformInfoWrapperPass>();
  au.setPreservesAll();
}

char FunctionInfoPass::ID = 0;

static RegisterPass<FunctionInfoPass> X("wlfuncinfo", "Function Info Extractor",
                                        true /* Only looks at CFG */,
                                        true /* Analysis Pass */);

}  // namespace cost
}  // namespace llvm
}  // namespace compy
