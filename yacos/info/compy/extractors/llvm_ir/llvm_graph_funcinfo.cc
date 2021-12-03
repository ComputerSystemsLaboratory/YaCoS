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

#include "llvm_graph_funcinfo.h"

#include <iostream>
#include <sstream>

#include "llvm/IR/Instructions.h"

using namespace ::llvm;

namespace compy {
namespace llvm {
namespace graph {

std::string llvmTypeToString(Type* type) {
    std::string typeName;
    raw_string_ostream rso(typeName);
    type->print(rso);
    return rso.str();
}

/**
 * Get a unique Name for an LLVM value.
 *
 * This function should always be used instead of the values getName()
 * function. If the object has no name yet, a new unique name is generated
 * based on the default name.
 */
std::string FunctionInfoPass::getUniqueName(const Value &v) {
  // if (v.hasName()) return v.getName(); (CHANGED)
  if (v.hasName()) return v.getName().str();

  auto iter = valueNames.find(&v);
  if (iter != valueNames.end()) return iter->second;

  std::stringstream ss;
  if (isa<Function>(v))
    ss << "func";
  else if (isa<BasicBlock>(v))
    ss << "bb";
  else if (isa<Value>(v))
    ss << "val";
  else
    ss << "v";

  ss << valueNames.size();

  valueNames[&v] = ss.str();
  return ss.str();
}

ArgInfoPtr FunctionInfoPass::getInfo(const Argument &arg) {
  auto it = argInfos.find(&arg);
  if (it != argInfos.end()) return it->second;

  ArgInfoPtr info(new ArgInfo());
  argInfos[&arg] = info;

  info->name = getUniqueName(arg);

  // collect the type
  info->type = llvmTypeToString(arg.getType());

  return info;
}

ConstantInfoPtr FunctionInfoPass::getInfo(const ::llvm::Constant &con) {
    auto it = constantInfos.find(&con);
    if (it != constantInfos.end()) return it->second;

    ConstantInfoPtr info(new ConstantInfo());
    constantInfos[&con] = info;

    // collect the type
    info->type = llvmTypeToString(con.getType());
    // collect the value
    // FIX IT: How to get the constant value???
    std::string value;
    raw_string_ostream sv(value);
    con.print(sv);
    std::size_t found = sv.str().find(" ");
    if (found)
       sv.str().erase(0, found+1);
    info->value = sv.str().find("Function Attrs:") ? sv.str() : "";

    return info;
}

InstructionInfoPtr FunctionInfoPass::getInfo(const Instruction &inst) {
  auto it = instructionInfos.find(&inst);
  if (it != instructionInfos.end()) return it->second;

  InstructionInfoPtr info(new InstructionInfo());
  instructionInfos[&inst] = info;

  // collect instruction (string)
  std::string instStr;
  raw_string_ostream ss(instStr);
  inst.print(ss);

  // Trim any leading indentation whitespace.
  // labm8::TrimLeft(str);
  // FIX IT: use lab8m
  std::size_t found = ss.str().find("  ");
  if (found == 0)
      ss.str().erase(0, 2);
  info->instStr = ss.str();

  // collect opcode
  info->opcode = inst.getOpcodeName();

  if (inst.getOpcodeName() == std::string("ret")) {
      info_->exitInstructions.push_back(info);
  }

  // collect type
  std::string typeName;
  raw_string_ostream rso(typeName);
  inst.getType()->print(rso);
  info->type = rso.str();

  // collect data dependencies
  for (auto &use : inst.operands()) {
      if (isa<Instruction>(use.get())) {
          auto &opInst = *cast<Instruction>(use.get());
          info->operands.push_back(getInfo(opInst));
      }

      if (isa<Argument>(use.get())) {
          auto &opInst = *cast<Argument>(use.get());
          info->operands.push_back(getInfo(opInst));
      }

      if (isa<Constant>(use.get())) {
          auto &opInst = *cast<Constant>(use.get());
          info->operands.push_back(getInfo(opInst));
      }
  }

  // collect called function (if this instruction is a call)
  if (isa<CallInst>(inst)) {
    auto &call = cast<CallInst>(inst);
    Function *calledFunction = call.getCalledFunction();
    if (calledFunction != nullptr) {
      info->callTarget = getUniqueName(*calledFunction);
    }
  }

  // load or store?
  info->isLoadOrStore = false;
  if (isa<LoadInst>(inst)) info->isLoadOrStore = true;
  if (isa<StoreInst>(inst)) info->isLoadOrStore = true;

  // collect function this instruction belongs to
  info->function = info_;

  return info;
}

BasicBlockInfoPtr FunctionInfoPass::getInfo(const BasicBlock &bb) {
  auto it = basicBlockInfos.find(&bb);
  if (it != basicBlockInfos.end()) return it->second;

  BasicBlockInfoPtr info(new BasicBlockInfo());
  basicBlockInfos[&bb] = info;

  info->name = getUniqueName(bb);

  // collect all successors
  auto term = bb.getTerminator();
  for (size_t i = 0; i < term->getNumSuccessors(); i++) {
    BasicBlock* succ = term->getSuccessor(i);
    info->successors.push_back(getInfo(*succ));
  }

  return info;
}

MemoryAccessInfoPtr FunctionInfoPass::getInfo(MemoryAccess &acc) {
  auto it = memoryAccessInfos.find(&acc);
  if (it != memoryAccessInfos.end()) return it->second;

  MemoryAccessInfoPtr info(new MemoryAccessInfo());
  memoryAccessInfos[&acc] = info;

  info->basicBlock = getInfo(*acc.getBlock());

  if (isa<MemoryUseOrDef>(acc)) {
    if (isa<MemoryUse>(acc))
      info->type = "use";
    else
      info->type = "def";

    auto inst = cast<MemoryUseOrDef>(acc).getMemoryInst();
    if (inst != nullptr) {
      info->inst = getInfo(*inst);
    } else {
      info->inst = NULL;
      assert(info->type == "def");
      info->type = "live on entry";
    }

    auto dep = cast<MemoryUseOrDef>(acc).getDefiningAccess();
    if (dep != nullptr) {
      info->dependencies.push_back(getInfo(*dep));
    }
  } else {
    info->type = "phi";
    info->inst = NULL;
    auto &phi = cast<MemoryPhi>(acc);
    for (unsigned i = 0; i < phi.getNumIncomingValues(); i++) {
      auto dep = phi.getIncomingValue(i);
      info->dependencies.push_back(getInfo(*dep));
    }
  }

  return info;
}

bool FunctionInfoPass::runOnFunction(::llvm::Function &func) {
  // wipe all data from the previous run
  valueNames.clear();
  argInfos.clear();
  basicBlockInfos.clear();
  instructionInfos.clear();
  memoryAccessInfos.clear();
  valueNames.clear();

  // create a new info object and invalidate the old one
  info_ = FunctionInfoPtr(new FunctionInfo());

  info_->name = getUniqueName(func);
  info_->entryInstruction = getInfo(*func.getEntryBlock().getInstList().begin());

  std::string rtypeName;
  raw_string_ostream rso(rtypeName);
  func.getReturnType()->print(rso);
  info_->type = rso.str();

  // Instruction cost
  auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(func);
  // Block edge frequency
  compy::llvm::wlcost::BlockEdgeFrequencyPass *BEF = &getAnalysis<compy::llvm::wlcost::BlockEdgeFrequencyPass>();

  // collect all basic blocks and their instructions
  for (auto &bb : func.getBasicBlockList()) {
    BasicBlockInfoPtr bbInfo = getInfo(bb);
    for (auto &inst : bb) {
      InstructionInfoPtr instInfo = getInfo(inst);
      instInfo->recipThroughput = TTI->getInstructionCost(
                                        &inst,
                                        TTI->TCK_RecipThroughput
                                  );
      instInfo->latency = TTI->getInstructionCost(
                                &inst,
                                TTI->TCK_Latency
                          );
      instInfo->codeSize = TTI->getInstructionCost(
                                &inst,
                                TTI->TCK_CodeSize
                          );
      instInfo->basicBlock = bbInfo;
      bbInfo->instructions.push_back(instInfo);
    }
    bbInfo->fullName = info_->name + "." + bbInfo->name;
    bbInfo->frequency = BEF->getBlockFrequency(&bb);
    info_->basicBlocks.push_back(bbInfo);
  }

  // collect all arguments
  for (auto &arg : func.args()) {
    info_->args.push_back(getInfo(arg));
  }

  // dump app memory accesses
  auto &mssaPass = getAnalysis<MemorySSAWrapperPass>();
  auto &mssa = mssaPass.getMSSA();
  for (auto &bb : func.getBasicBlockList()) {
    // live on entry
    auto entry = mssa.getLiveOnEntryDef();
    info_->memoryAccesses.push_back(getInfo(*entry));

    // memory phis
    auto phi = mssa.getMemoryAccess(&bb);
    if (phi != nullptr) {
      info_->memoryAccesses.push_back(getInfo(*phi));
    }

    // memory use or defs
    for (auto &inst : bb) {
      auto access = mssa.getMemoryAccess(&inst);
      if (access != nullptr) {
        info_->memoryAccesses.push_back(getInfo(*access));
      }
    }
  }

  // indicate that nothing was changed
  return false;
}

void FunctionInfoPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<MemorySSAWrapperPass>();
  au.addRequired<TargetTransformInfoWrapperPass>();
  au.addRequired<compy::llvm::wlcost::BlockEdgeFrequencyPass>();
  au.setPreservesAll();
}

char FunctionInfoPass::ID = 0;

static RegisterPass<FunctionInfoPass> X("funcinfo", "Function Info Extractor",
                                        true /* Only looks at CFG */,
                                        true /* Analysis Pass */);

}  // namespace graph
}  // namespace llvm
}  // namespace compy
