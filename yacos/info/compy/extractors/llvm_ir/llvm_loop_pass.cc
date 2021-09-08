
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

// Wu Larus function cost

#include <algorithm>
#include <string>
#include <iostream>

#include "llvm_loop_pass.h"
#include "llvm_loop_funcinfo.h"

using namespace ::llvm;

namespace compy {
namespace llvm {
namespace loop {

bool ExtractorPass::runOnModule(::llvm::Module &module) {
  ExtractionInfoPtr info(new ExtractionInfo);

  // Collect the function information
  for (auto &func : module.functions()) {
    // Skip functions without definition (fwd declarations)
    if (func.isDeclaration())
      continue;

    auto &pass = getAnalysis<FunctionInfoPass>(func);
    auto functionInfo = std::move(pass.getInfo());
    info->functionInfos.push_back(std::move(functionInfo));
  }

  this->extractionInfo = info;
  return false;
}

void ExtractorPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<FunctionInfoPass>();
  au.setPreservesAll();
}

char ExtractorPass::ID = 0;
static ::llvm::RegisterPass<ExtractorPass> X("loopExtractor", "Loop Features Extractor",
                                             true /* Only looks at CFG */,
                                             true /* Analysis Pass */);

}  // namespace loop´
}  // namespace llvm
}  // namespace compy
