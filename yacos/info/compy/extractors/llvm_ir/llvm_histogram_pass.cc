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

#include "llvm_histogram_pass.h"

#include <algorithm>
#include <string>
#include <iostream>
#include <set>

#include "llvm/Analysis/CallGraph.h"
#include <llvm/Analysis/CFG.h>
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Value.h"

#include "llvm/Transforms/IPO.h"

using namespace ::llvm;

namespace compy {
namespace llvm {
namespace histogram {

void ExtractorPass::initInstructions(){
  if (!this->instructions.empty())
    this->instructions.clear();

  this->updateInstruction("ret", 0.0);
  this->updateInstruction("br", 0.0);
  this->updateInstruction("switch", 0.0);
  this->updateInstruction("indirectbr", 0.0);
  this->updateInstruction("invoke", 0.0);
  this->updateInstruction("callbr", 0.0);
  this->updateInstruction("resume", 0.0);
  this->updateInstruction("catchswitch", 0.0);
  this->updateInstruction("catchret", 0.0);
  this->updateInstruction("cleanupret", 0.0);
  this->updateInstruction("unreachable", 0.0);
  this->updateInstruction("fneg", 0.0);
  this->updateInstruction("add", 0.0);
  this->updateInstruction("fadd", 0.0);
  this->updateInstruction("sub", 0.0);
  this->updateInstruction("fsub", 0.0);
  this->updateInstruction("mul", 0.0);
  this->updateInstruction("fmul", 0.0);
  this->updateInstruction("udiv", 0.0);
  this->updateInstruction("sdiv", 0.0);
  this->updateInstruction("fdiv", 0.0);
  this->updateInstruction("urem", 0.0);
  this->updateInstruction("srem", 0.0);
  this->updateInstruction("frem", 0.0);
  this->updateInstruction("shl", 0.0);
  this->updateInstruction("lshr", 0.0);
  this->updateInstruction("ashr", 0.0);
  this->updateInstruction("and", 0.0);
  this->updateInstruction("or", 0.0);
  this->updateInstruction("xor", 0.0);
  this->updateInstruction("extractelement", 0.0);
  this->updateInstruction("insertelement", 0.0);
  this->updateInstruction("sufflevector", 0.0);
  this->updateInstruction("extractvalue", 0.0);
  this->updateInstruction("insertvalue", 0.0);
  this->updateInstruction("alloca", 0.0);
  this->updateInstruction("load", 0.0);
  this->updateInstruction("store", 0.0);
  this->updateInstruction("fence", 0.0);
  this->updateInstruction("cmpxchg", 0.0);
  this->updateInstruction("atomicrmw", 0.0);
  this->updateInstruction("getelementptr", 0.0);
  this->updateInstruction("trunc", 0.0);
  this->updateInstruction("zext", 0.0);
  this->updateInstruction("sext", 0.0);
  this->updateInstruction("fptrunc", 0.0);
  this->updateInstruction("fpext", 0.0);
  this->updateInstruction("fptoui", 0.0);
  this->updateInstruction("fptosi", 0.0);
  this->updateInstruction("uitofp", 0.0);
  this->updateInstruction("sitofp", 0.0);
  this->updateInstruction("ptrtoint", 0.0);
  this->updateInstruction("inttoptr", 0.0);
  this->updateInstruction("bitcast", 0.0);
  this->updateInstruction("addrspacecast", 0.0);
  this->updateInstruction("icmp", 0.0);
  this->updateInstruction("fcmp", 0.0);
  this->updateInstruction("phi", 0.0);
  this->updateInstruction("select", 0.0);
  this->updateInstruction("freeze", 0.0);
  this->updateInstruction("call", 0.0);
  this->updateInstruction("var_arg", 0.0);
  this->updateInstruction("landingpad", 0.0);
  this->updateInstruction("catchpad", 0.0);
  this->updateInstruction("cleanuppad", 0.0);
}

void ExtractorPass::updateInstruction(std::string  key, float value){
  std::map<std::string, float>::iterator it = this->instructions.find(key);
  if (it == this->instructions.end())
    this->instructions[key] = value;
  else
    it->second += value;
}

bool ExtractorPass::runOnModule(::llvm::Module &module) {
  ExtractionInfoPtr info(new ExtractionInfo);

  for (const auto &F : module.functions()) {
    if (F.isDeclaration())
      continue;

    this->initInstructions();

    for (const auto &bb : F.getBasicBlockList())
      for (const auto &inst : bb)
        this->updateInstruction(inst.getOpcodeName(), 1.0);

    FunctionInfoPtr functionInfo(new FunctionInfo);
    functionInfo->name = F.getName().str();
    functionInfo->instructions = this->instructions;
    info->functionInfos.push_back(functionInfo);
  }

  this->extractionInfo = info;
  return false;
}

void ExtractorPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.setPreservesAll();
}

char ExtractorPass::ID = 0;
static ::llvm::RegisterPass<ExtractorPass> X("histogramExtractor", "HistogramExtractor",
                                              true /* Only looks at CFG */,
                                              true /* Analysis Pass */);

}  // namespace histogram
}  // namespace llvm
}  // namespace compy
