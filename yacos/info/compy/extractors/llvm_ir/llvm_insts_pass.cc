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

#include "llvm_insts_pass.h"

#include <algorithm>
#include <string>
#include <iostream>

using namespace ::llvm;

namespace compy {
namespace llvm {
namespace insts {

bool ExtractorPass::runOnModule(::llvm::Module &module) {
  ExtractionInfoPtr info(new ExtractionInfo);

  for (const auto &F : module.functions()) {
    if (F.isDeclaration())
      continue;

    unsigned instructions = 0;
    for (const auto &bb : F.getBasicBlockList())
      for (const auto &inst : bb)
        instructions++;

    FunctionInfoPtr functionInfo(new FunctionInfo);
    functionInfo->name = F.getName().str();
    functionInfo->instructions = instructions;

    info->functionInfos.push_back(functionInfo);
  }

  this->extractionInfo = info;
  return false;
}

void ExtractorPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.setPreservesAll();
}

char ExtractorPass::ID = 0;
static ::llvm::RegisterPass<ExtractorPass> X("instsExtractor", "InstsExtractor",
                                              true /* Only looks at CFG */,
                                              true /* Analysis Pass */);

}  // namespace insts
}  // namespace llvm
}  // namespace compy
