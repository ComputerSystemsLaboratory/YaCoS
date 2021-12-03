
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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/DependenceAnalysis.h"

#include "llvm_extractor.h"

namespace compy {
namespace llvm {
namespace loop {

class FunctionInfoPass : public ::llvm::FunctionPass {
 private:
  FunctionInfoPtr info_;

  std::vector<int> tracking_;
  std::map<std::string, float> features_;

  void initFeatures();
  void updateFeature(std::string key, float value);
  void DependenceCheckFunctionIntern(const ::llvm::Function &F,
                                    ::llvm::DependenceInfo &DI,
                                    std::vector<::llvm::Instruction*> loopInsts);
  void DependenceCheckFunctionExtern(const ::llvm::Function &F,
                                    ::llvm::DependenceInfo &DI,
                                    std::vector<::llvm::Instruction*> loopInsts);
  void RecursiveIterLoopFramework(const ::llvm::Function &F,
                                 ::llvm::DependenceInfo &DI,
                                 ::llvm::Loop *L,
                                 std::vector<::llvm::Instruction*> defUseOfArrays,
                                 unsigned nesting);
  void FeaturesExtractor(::llvm::Function &F, ::llvm::DependenceInfo &DI, ::llvm::LoopInfo &LI);
 public:
  static char ID;

  FunctionInfoPass() : ::llvm::FunctionPass(ID), info_(nullptr) {}

  bool runOnFunction(::llvm::Function &func) override;
  void getAnalysisUsage(::llvm::AnalysisUsage &au) const override;

  const FunctionInfoPtr &getInfo() const { return info_; }
  FunctionInfoPtr &getInfo() { return info_; }
};

}  // namespace loop
}  // namespace llvm
}  // namespace compy
