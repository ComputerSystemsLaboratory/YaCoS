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

#include "llvm_msf_pass.h"

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
namespace msf {

void ExtractorPass::initFeatures(){
  if (!this->features.empty())
    this->features.clear();
  this->updateFeature("ft01_BBInMethod", 0.0);
  this->updateFeature("ft02_BBWithOneSuccessor", 0.0);
  this->updateFeature("ft03_BBWithTwoSuccessors", 0.0);
  this->updateFeature("ft04_BBWithMoreThanTwoSuccessors", 0.0);
  this->updateFeature("ft05_BBWithOnePredecessor", 0.0);
  this->updateFeature("ft06_BBWithTwoPredecessors", 0.0);
  this->updateFeature("ft07_BBWithMoreThanTwoPredecessors", 0.0);
  this->updateFeature("ft08_BBWithOnePredOneSuc", 0.0);
  this->updateFeature("ft09_BBWithOnePredTwoSuc", 0.0);
  this->updateFeature("ft10_BBWithTwoPredOneSuc", 0.0);
  this->updateFeature("ft11_BBWithTwoPredTwoSuc", 0.0);
  this->updateFeature("ft12_BBWithMoreTwoPredMoreTwoSuc", 0.0);
  this->updateFeature("ft13_BBWithInstructionsLessThan15", 0.0);
  this->updateFeature("ft14_BBWithInstructionsIn[15-500]", 0.0);
  this->updateFeature("ft15_BBWithInstructionsGreaterThan500", 0.0);
  this->updateFeature("ft16_EdgesInCFG", 0.0);
  this->updateFeature("ft17_CriticalEdgesInCFG", 0.0);
  this->updateFeature("ft18_AbnormalEdgesInCFG", 0.0);
  this->updateFeature("ft19_DirectCalls", 0.0);
  this->updateFeature("ft20_ConditionalBranch", 0.0);
  this->updateFeature("ft21_AssignmentInstructions", 0.0);
  this->updateFeature("ft22_ConditionalBranch", 0.0);
  this->updateFeature("ft23_BinaryIntOperations", 0.0);
  this->updateFeature("ft24_BinaryFloatPTROperations", 0.0);
  this->updateFeature("ft25_Instructions", 0.0);
  this->updateFeature("ft26_AverageInstruction", 0.0);
  this->updateFeature("ft27_AveragePhiNodes", 0.0);
  this->updateFeature("ft28_AverageArgsPhiNodes", 0.0);
  this->updateFeature("ft29_BBWithoutPhiNodes", 0.0);
  this->updateFeature("ft30_BBWithPHINodesIn[0-3]", 0.0);
  this->updateFeature("ft31_BBWithMoreThan3PHINodes", 0.0);
  this->updateFeature("ft32_BBWithArgsPHINodesGreaterThan5", 0.0);
  this->updateFeature("ft33_BBWithArgsPHINodesGreaterIn[1-5]", 0.0);
  this->updateFeature("ft34_SwitchInstructions", 0.0);
  this->updateFeature("ft35_UnaryOperations", 0.0);
  this->updateFeature("ft36_InstructionThatDoPTRArithmetic", 0.0);
  this->updateFeature("ft37_IndirectRefs", 0.0);
  this->updateFeature("ft38_AdressVarIsTaken", 0.0);
  this->updateFeature("ft39_AddressFunctionIsTaken", 0.0);
  this->updateFeature("ft40_IndirectCalls", 0.0);
  this->updateFeature("ft41_AssignmentInstructionsWithLeftOperandIntegerConstant", 0.0);
  this->updateFeature("ft42_BinaryOperationsWithOneOperandIntegerConstant", 0.0);
  this->updateFeature("ft43_CallsWithPointersArgument", 0.0);
  this->updateFeature("ft44_CallsWithArgsGreaterThan4", 0.0);
  this->updateFeature("ft45_CallsThatReturnPTR", 0.0);
  this->updateFeature("ft46_CallsThatReturnInt", 0.0);
  this->updateFeature("ft47_ConstantZero", 0.0);
  this->updateFeature("ft48_32-bitIntegerConstants", 0.0);
  this->updateFeature("ft49_ConstantOne", 0.0);
  this->updateFeature("ft50_64-bitIntegerConstants", 0.0);
  this->updateFeature("ft51_ReferencesLocalVariables", 0.0);
  this->updateFeature("ft52_DefUseVariables", 0.0);
  this->updateFeature("ft53_LocalVariablesReferred", 0.0);
  this->updateFeature("ft54_ExternVariablesReferred", 0.0);
  this->updateFeature("ft55_LocalVariablesPointers", 0.0);
  this->updateFeature("ft56_VariablesPointers", 0.0);
}

void ExtractorPass::updateFeature(std::string  key, float value){
  std::map<std::string, float>::iterator it = this->features.find(key);
  if (it == this->features.end())
    this->features[key] = value;
  else
    it->second += value;
}

bool ExtractorPass::runOnModule(::llvm::Module &module) {
  ExtractionInfoPtr info(new ExtractionInfo);

  for (const auto &F : module.functions()) {
    if (F.isDeclaration())
      continue;

    unsigned localPhiArgs = 0;
    unsigned totalPhiArgs = 0;
    unsigned phiCounterBlock = 0;
    unsigned totalPhi = 0;
    unsigned blockWithPhi = 0;
    std::set<Value *> varRefTemp;
    std::set<GlobalVariable *> ExterRefVarTemp;

    this->initFeatures();

    for (const auto &bb : F.getBasicBlockList()) {
      unsigned numOfSuccessors = 0;
      for (auto it = succ_begin(&bb), et = succ_end(&bb); it != et; ++it)
        ++numOfSuccessors;

      unsigned numOfPredecessors = 0;
      for (auto it = pred_begin(&bb), et = pred_end(&bb); it != et; ++it)
        ++numOfPredecessors;

      if (numOfSuccessors == 1)
        this->updateFeature("ft02_BBWithOneSuccessor", 1.0);
      else if (numOfSuccessors == 2)
        this->updateFeature("ft03_BBWithTwoSuccessors", 1.0);
      else if (numOfSuccessors > 2)
        this->updateFeature("ft04_BBWithMoreThanTwoSuccessors", 1.0);

      if (numOfPredecessors == 1)
        this->updateFeature("ft05_BBWithOnePredecessor", 1.0);
      else if (numOfPredecessors == 2)
        this->updateFeature("ft06_BBWithTwoPredecessors", 1.0);
      else if (numOfPredecessors > 2)
        this->updateFeature("ft07_BBWithMoreThanTwoPredecessors", 1.0);

      if ((numOfPredecessors == 1) && (numOfSuccessors == 1))
        this->updateFeature("ft08_BBWithOnePredOneSuc", 1.0);
      else if ((numOfPredecessors == 1) && (numOfSuccessors == 2))
        this->updateFeature("ft09_BBWithOnePredTwoSuc", 1.0);
      else if ((numOfPredecessors == 2) && (numOfSuccessors == 1))
        this->updateFeature("ft10_BBWithTwoPredOneSuc", 1.0);
      else if ((numOfPredecessors == 2) && (numOfSuccessors == 2))
        this->updateFeature("ft11_BBWithTwoPredTwoSuc", 1.0);
      else if ((numOfPredecessors > 2) && (numOfSuccessors > 2))
        this->updateFeature("ft12_BBWithMoreTwoPredMoreTwoSuc", 1.0);

      this->updateFeature("ft16_EdgesInCFG", (float)(numOfSuccessors+numOfPredecessors));

      if (numOfSuccessors != 0)
        for (unsigned m = 0; m < numOfSuccessors; m++)
          if (isCriticalEdge(bb.getTerminator(), m))
            this->updateFeature("ft17_CriticalEdgesInCFG", 1.0);

      this->updateFeature("ft01_BBInMethod", 1.0);

      for (const auto &inst : bb) {

  	    unsigned numOperands = inst.getNumOperands();
  	    unsigned tempIndirecCount = 0;
  	    Type *instTy = inst.getType();

  	    if (/*auto *SI = */dyn_cast<StoreInst>(&inst)) {
  	      /*
            There are two arguments to the store instruction:
  		      a value to store and an address at which to store it
  	      */
  	      Value *isInstValue = inst.getOperand(0);
  	      Type *getTypeBits = isInstValue->getType();

  	      if (getTypeBits->isPointerTy())
  		        this->updateFeature("ft37_IndirectRefs", 1.0);
  	      if (isa<Function>(isInstValue))
  		        this->updateFeature("ft39_AddressFunctionIsTaken", 1.0);
  	      if ((getTypeBits->isPointerTy()) && (isa<Instruction>(isInstValue)))
  		        this->updateFeature("ft38_AdressVarIsTaken", 1.0);

  	    } else if (auto *BI = dyn_cast<BranchInst>(&inst)) {
    	      if (BI->isConditional())
    		      this->updateFeature("ft20_ConditionalBranch", 1.0);
    	      else if (BI->isUnconditional())
    		      this->updateFeature("ft22_ConditionalBranch", 1.0);
  	    } else if (/*auto *SI = */dyn_cast<SwitchInst>(&inst)) {
    	       this->updateFeature("ft34_SwitchInstructions", 1.0);
  	    } else if (auto *CI = dyn_cast<CallInst>(&inst)) {
    	      Function *callFunction = CI->getCalledFunction();
    	      Type *callType = CI->getType();

    	      if (callType->isPointerTy())
    		      this->updateFeature("ft45_CallsThatReturnPTR", 1.0);
            else if (callType->isIntegerTy())
                this->updateFeature("ft46_CallsThatReturnInt", 1.0);
    	      /*
              if calledFunction is nullptr and stripped value is a
    		      function, then, it's a direct call in the generate assembly.
    		      (Ref:
    		      https://lists.llvm.org/pipermail/llvm-dev/2018-August/125098.html)
    	      */
    	      if (callFunction == nullptr) {
    			    if (CI->isIndirectCall())
    		        this->updateFeature("ft40_IndirectCalls", 1.0);
    		      else
    		        this->updateFeature("ft19_DirectCalls", 1.0);
    	      } else
    		      this->updateFeature("ft19_DirectCalls", 1.0);

    	      unsigned argsCount = 0;
    	      if (callFunction != nullptr) {
    		        unsigned numArgOp = CI->getNumArgOperands();
    		        for (unsigned arg = 0; arg < numArgOp; arg++) {
    		            Type *argTy = CI->getArgOperand(arg)->getType();
    		            if (argTy->isPointerTy())
    		              ++tempIndirecCount;
    		            ++argsCount;
    		        }
            }

    	      if (tempIndirecCount != 0)
    		      this->updateFeature("ft43_CallsWithPointersArgument", 1.0);
    	      if (argsCount > 4)
    		      this->updateFeature("ft44_CallsWithArgsGreaterThan4", 1.0);
  	    }
  	    /*  The ‘invoke’ instruction causes control to transfer to a specified
  		    function, with the possibility of control flow transfer to either
  		    the ‘normal’ label or the ‘exception’ label. If the callee function
  		    returns with the “ret” instruction, control flow will return to the
  		    “normal” label. If the callee (or any indirect callees) returns via
  		    the “resume” instruction or other exception handling mechanism,
  		    control is interrupted and continued at the dynamically nearest
  		    “exception” label. (Ref:
  		    http://llvm.org/docs/LangRef.html#invoke-instruction)
  	    */
  	    else if (/*auto *II = */dyn_cast<InvokeInst>(&inst))
          this->updateFeature("ft18_AbnormalEdgesInCFG", 1.0);

        const UnaryOperator *UN = dyn_cast<UnaryOperator>(&inst);
        const BinaryOperator *BIO = dyn_cast<BinaryOperator>(&inst);

  	    if (UN)
  	      this->updateFeature("ft35_UnaryOperations", 1.0);
  	    else if (BIO) {

  	      Value *firstBinOp = BIO->getOperand(0);
  	      Value *secondBinOp = BIO->getOperand(1);
  	      Type *firstBinOpTy = BIO->getOperand(0)->getType();
  	      Type *secondBinOpTy = BIO->getOperand(1)->getType();

  	      if ((firstBinOpTy->isIntegerTy()) && (secondBinOpTy->isIntegerTy()))
  		      this->updateFeature("ft23_BinaryIntOperations", 1.0);
  	      else if ((firstBinOpTy->isFloatingPointTy()) && (secondBinOpTy->isFloatingPointTy()))
  		      this->updateFeature("ft24_BinaryFloatPTROperations", 1.0);
  	      else if ((firstBinOpTy->isPointerTy()) || (secondBinOpTy->isPointerTy()))
  		      this->updateFeature("ft36_InstructionThatDoPTRArithmetic", 1.0);

  	      if ((isa<ConstantInt>(firstBinOp)) || (isa<ConstantInt>(secondBinOp)))
  		      this->updateFeature("ft42_BinaryOperationsWithOneOperandIntegerConstant", 1.0);
  	    }

  	    if (isa<PHINode>(&inst)) {
    	      /* The num of operands is the num of arguments for a phi node*/
    	      unsigned phiOperands = inst.getNumOperands();
    	      localPhiArgs += phiOperands;
    	      totalPhiArgs += phiOperands;
    	      ++phiCounterBlock;
    	      ++totalPhi;
  	    }

  	    if ((instTy->isVoidTy()) == 0) {
  	      this->updateFeature("ft21_AssignmentInstructions", 1.0);
  	      /* Number of assignment instructions with the left operand
  		      an integer constant in the method. Here, a left operand is
  		      being considered as the first operand in instruction with
  		      two operands. */
  	      if (numOperands == 2)
  		      if (/*ConstantInt *intConst = */dyn_cast<ConstantInt>(inst.getOperand(0)))
  		        this->updateFeature("ft41_AssignmentInstructionsWithLeftOperandIntegerConstant", 1.0);

  	      if (instTy->isPointerTy())
  		      this->updateFeature("ft55_LocalVariablesPointers", 1.0);
  	    }

  	    if (isa<AllocaInst>(&inst) == 0) {
  	       for (const Use &U : inst.operands()) {
             if (isa<Instruction>(U)) {
  		          varRefTemp.insert(U);
  		          this->updateFeature("ft51_ReferencesLocalVariables", 1.0);
  		       }
  		       if (ConstantInt *zeroOne = dyn_cast<ConstantInt>(U)) {
          		  Type *getTypeBits = zeroOne->getType();

          		  if (getTypeBits->isIntegerTy(32))
          		    this->updateFeature("ft48_32-bitIntegerConstants", 1.0);
          		  else if (getTypeBits->isIntegerTy(64))
          		    this->updateFeature("ft50_64-bitIntegerConstants", 1.0);
          		  if ((zeroOne->getSExtValue()) == 0 && (getTypeBits->isIntegerTy()))
          		    this->updateFeature("ft47_ConstantZero", 1.0);
          		  else if ((zeroOne->getSExtValue() == 1) && (getTypeBits->isIntegerTy()))
          		    this->updateFeature("ft49_ConstantOne", 1.0);
  		       }
  		       if (GlobalVariable *GV = dyn_cast<GlobalVariable>(U)) {
          		  ExterRefVarTemp.insert(GV);
          		  if (GV->getNumOperands()) {
          		    Type *getType = GV->getOperand(0)->getType();
          		    if (getType->isPointerTy())
          		      this->updateFeature("f56_VariablesPointers", 1.0);
          		  }
          		  this->updateFeature("ft52_DefUseVariables", 1.0);
          	}
          }
        }

        this->updateFeature("ft25_Instructions", 1.0);

      }

      if (phiCounterBlock > 3)
          this->updateFeature("ft31_BBWithMoreThan3PHINodes", 1.0);
      else if (/*(phiCounterBlock >= 0) && */(phiCounterBlock <= 3))
          this->updateFeature("ft30_BBWithPHINodesIn[0-3]", 1.0);

      if (localPhiArgs > 5)
          this->updateFeature("ft32_BBWithArgsPHINodesGreaterThan5", 1.0);
      else if ((localPhiArgs > 0) && (localPhiArgs < 6))
          this->updateFeature("ft33_BBWithArgsPHINodesGreaterIn[1-5]", 1.0);

      if (phiCounterBlock != 0)
          ++blockWithPhi;

      unsigned instPerBB = bb.size();
      if (instPerBB < 15)
          this->updateFeature("ft13_BBWithInstructionsLessThan15", 1.0);
      else if ((instPerBB >= 15) && (instPerBB <= 500))
          this->updateFeature("ft14_BBWithInstructionsIn[15-500]", 1.0);
      else if (instPerBB > 500)
          this->updateFeature("ft15_BBWithInstructionsGreaterThan500", 1.0);
      }

      std::map<std::string, float>::iterator it = this->features.find("ft01_BBInMethod");
      float totalBasicBlocks = it->second;

      it = this->features.find("ft25_Instructions");
      float numInstructions = it->second;

      this->updateFeature("ft26_AverageInstruction", (float)(numInstructions / totalBasicBlocks));
      this->updateFeature("ft27_AveragePhiNodes", (float)(totalPhi / totalBasicBlocks));

      if (totalPhiArgs)
        this->updateFeature("ft28_AverageArgsPhiNodes", (float)(totalPhiArgs / totalPhi));

      this->updateFeature("ft53_LocalVariablesReferred", (float)varRefTemp.size());
      this->updateFeature("ft54_ExternVariablesReferred", (float)ExterRefVarTemp.size());
      this->updateFeature("ft29_BBWithoutPhiNodes", (float)(totalBasicBlocks - blockWithPhi));

      FunctionInfoPtr functionInfo(new FunctionInfo);
      functionInfo->name = F.getName().str();
      functionInfo->features = this->features;
      info->functionInfos.push_back(functionInfo);
  }

  this->extractionInfo = info;
  return false;
}

void ExtractorPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.setPreservesAll();
}

char ExtractorPass::ID = 0;
static ::llvm::RegisterPass<ExtractorPass> X("msfExtractor", "MSFExtractor",
                                              true /* Only looks at CFG */,
                                              true /* Analysis Pass */);

}  // namespace msf
}  // namespace llvm
}  // namespace compy
