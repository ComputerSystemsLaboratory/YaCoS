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

#include "llvm_ir2vec_pass.h"
#include "IR2Vec.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <sys/stat.h>

using namespace ::llvm;

namespace compy {
namespace llvm {
namespace ir2vec {

bool file_exists (const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

bool ExtractorPass::runOnModule(::llvm::Module &module) {
  ExtractionInfoPtr info(new ExtractionInfo);

  const char* env_p = std::getenv("HOME");
  std::string vocabulary;
  vocabulary = env_p;
  vocabulary = vocabulary + "/.local/yacos/data/ir2vec/seedEmbeddingVocab-300-llvm10.txt";

  if (!file_exists(vocabulary))
      throw std::runtime_error("YaCoS data does not exist.");

  auto ir2vec = IR2Vec::Embeddings(module, IR2Vec::IR2VecMode::FlowAware,
                                   vocabulary);

  // Getting Instruction vectors corresponding to the
  // instructions in LLVM module
  auto instVecMap = ir2vec.getInstVecMap();
  // Access the generated vectors
  for (auto instVec : instVecMap) {
      InstructionInfoPtr instructionInfo(new InstructionInfo);

      // collect instruction (string)
      std::string instStr;
      raw_string_ostream ss(instStr);
      instVec.first->print(ss);
      // Trim any leading indentation whitespace.
      // labm8::TrimLeft(str);
      // FIX IT: use lab8m
      std::size_t found = ss.str().find("  ");
      if (found == 0)
          ss.str().erase(0, 2);
      instructionInfo->instStr = ss.str();

      // collect opcode
      instructionInfo->opcode = instVec.first->getOpcodeName();

      for (auto val : instVec.second)
        instructionInfo->ir2vec.push_back(val);
      info->instructionInfos.push_back(instructionInfo);
  }

  // Getting vectors corresponding to the functions in <LLVM Module>
  auto funcVecMap = ir2vec.getFunctionVecMap();
  // Access the generated vectors
  for (auto funcVec : funcVecMap) {
    FunctionInfoPtr functionInfo(new FunctionInfo);
    functionInfo->name = funcVec.first->getName();
    for (auto val : funcVec.second)
      functionInfo->ir2vec.push_back(val);
    info->functionInfos.push_back(functionInfo);
  }

  // Getting the program vector
  auto pgmVec = ir2vec.getProgramVector();
  // Access the generated vector
  ModuleInfoPtr moduleInfo(new ModuleInfo);
  moduleInfo->name = module.getName();
  for (auto val : pgmVec)
      moduleInfo->ir2vec.push_back(val);
  info->moduleInfo = moduleInfo;

  this->extractionInfo = info;
  return false;
}

void ExtractorPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.setPreservesAll();
}

char ExtractorPass::ID = 0;
static ::llvm::RegisterPass<ExtractorPass> X("ir2vecExtractor", "IR2Vec Extractor",
                                             true /* Only looks at CFG */,
                                             true /* Analysis Pass */);

}  // namespace names
}  // namespace llvm
}  // namespace compy
