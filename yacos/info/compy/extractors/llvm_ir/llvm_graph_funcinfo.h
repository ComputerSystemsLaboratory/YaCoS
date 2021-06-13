/*
Copyright 2020 Alexander Brauckmann

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

#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "wl_static_profiler/block_edge_frequency_pass.h"
#include "llvm_extractor.h"

namespace compy {
namespace llvm {
namespace graph {

class FunctionInfoPass : public ::llvm::FunctionPass {
 private:
  FunctionInfoPtr info_;

 public:
  static char ID;

  FunctionInfoPass() : ::llvm::FunctionPass(ID), info_(nullptr) {}

  bool runOnFunction(::llvm::Function &func) override;
  void getAnalysisUsage(::llvm::AnalysisUsage &au) const override;

  const FunctionInfoPtr &getInfo() const { return info_; }
  FunctionInfoPtr &getInfo() { return info_; }

 private:
  std::string getUniqueName(const ::llvm::Value &v);
  ArgInfoPtr getInfo(const ::llvm::Argument &arg);
  ConstantInfoPtr getInfo(const ::llvm::Constant &con);
  BasicBlockInfoPtr getInfo(const ::llvm::BasicBlock &bb);
  InstructionInfoPtr getInfo(const ::llvm::Instruction &inst);
  MemoryAccessInfoPtr getInfo(::llvm::MemoryAccess &acc);

 private:
  std::unordered_map<const ::llvm::Argument *, ArgInfoPtr> argInfos;
  std::unordered_map<const ::llvm::Constant *, ConstantInfoPtr> constantInfos;
  std::unordered_map<const ::llvm::BasicBlock *, BasicBlockInfoPtr>
      basicBlockInfos;
  std::unordered_map<const ::llvm::Instruction *, InstructionInfoPtr>
      instructionInfos;
  std::unordered_map<const ::llvm::MemoryAccess *, MemoryAccessInfoPtr>
      memoryAccessInfos;
  std::unordered_map<const ::llvm::Value *, std::string> valueNames;
};

}  // namespace graph
}  // namespace llvm
}  // namespace compy
