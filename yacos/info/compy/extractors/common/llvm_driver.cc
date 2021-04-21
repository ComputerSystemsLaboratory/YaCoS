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

#include "llvm_driver.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Config/config.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticBuffer.h"
#include "clang/FrontendTool/Utils.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"

using namespace ::clang;
using namespace ::llvm;

namespace compy {

constexpr unsigned int encodePassName(const char* str, int h=0) {
    return !str[h] ? 5381 : (encodePassName(str, h+1) * 33) ^ str[h];
}

LLVMDriver::LLVMDriver(std::vector<std::string> optimizations)
    : optimizations_(optimizations) {}

LLVMDriver::LLVMDriver()
    : optimizations_({}) {}

void LLVMDriver::addOptimizationFront(std::string optimization) {
  optimizations_.insert(optimizations_.begin(), optimization);
}

void LLVMDriver::addOptimizationsFront(std::vector<std::string> optimizations) {
  for (auto it = optimizations.rbegin(); it != optimizations.rend(); ++it)
    optimizations_.insert(optimizations_.begin(), *it);
}

void LLVMDriver::addOptimizationBack(std::string optimization) {
  optimizations_.insert(optimizations_.end(), optimization);
}

void LLVMDriver::addOptimizationsBack(std::vector<std::string> optimizations) {
  for (std::string optimization : optimizations)
    optimizations_.insert(optimizations_.end(), optimization);
}

void LLVMDriver::removeOptimization(std::string optimization) {
  optimizations_.erase(std::remove(optimizations_.begin(), optimizations_.end(),
                                  optimization),
                      optimizations_.end());
}

void LLVMDriver::removeOptimizationsFront(int optimizations) {
   optimizations_.erase(optimizations_.begin(), optimizations_.begin()+optimizations);
}

void LLVMDriver::removeOptimizationsBack(int optimizations) {
   optimizations_.erase(optimizations_.end()-optimizations, optimizations_.end());
}

void LLVMDriver::clearOptimizations() {
  optimizations_.clear();
}

void LLVMDriver::setOptimizations(std::vector<std::string> optimizations) {
  optimizations_ = optimizations;
}

std::vector<std::string> LLVMDriver::getOptimizations() {
  return optimizations_;
}

void LLVMDriver::addOptimizationPasses() {
 for (std::string pass : optimizations_)
    switch (encodePassName(pass.c_str())) {

    // "Transform/Scalar.h"
    case encodePassName("-constprop"): // Simple constant propagation
      pm_->add(llvm::createConstantPropagationPass()); // FunctionPass
      break;
    case encodePassName("-alignment-from-assumptions"): // Alignment from assumptions
      pm_->add(llvm::createAlignmentFromAssumptionsPass()); // FunctionPass
      break;
    case encodePassName("-sccp"): // Sparse Conditional Constant Propagation
      pm_->add(llvm::createSCCPPass()); // FunctionPass
      break;
    case encodePassName("-die"): // Dead Instruction Elimination
      pm_->add(llvm::createDeadInstEliminationPass()); // Pass
      break;
    case encodePassName("-redundant-dbg-inst-elim"): //  Redundant Dbg Instruction Elimination
      pm_->add(llvm::createRedundantDbgInstEliminationPass()); // Pass
      break;
    case encodePassName("-dce"): // Dead Code Elimination
      pm_->add(llvm::createDeadCodeEliminationPass()); // FunctionPass
      break;
    case encodePassName("-dse"): // Dead Store Elimination
      pm_->add(llvm::createDeadStoreEliminationPass()); // FunctionPass
      break;
    case encodePassName("-callsite-splitting"): // Call-site splitting
      pm_->add(llvm::createCallSiteSplittingPass()); // FunctionPass
      break;
    case encodePassName("-adce"): // Aggressive Dead Code Elimination
      pm_->add(llvm::createAggressiveDCEPass()); // FunctionPass
      break;
    case encodePassName("-guard-widening"): // Widen guards
      pm_->add(llvm::createGuardWideningPass()); // FunctionPass
      break;
    case encodePassName("-loop-guard-widening"): // Widen guards (within a single loop, as a loop pass)
      pm_->add(llvm::createLoopGuardWideningPass()); // Pass
      break;
    case encodePassName("-bdce"): // Bit-Tracking Dead Code Elimination
      pm_->add(llvm::createBitTrackingDCEPass()); // FunctionPass
      break;
    case encodePassName("-sroa"): // Scalar Replacement Of Aggregates
      pm_->add(llvm::createSROAPass()); // FunctionPass
      break;
    case encodePassName("-irce"): // Inductive range check elimination
      pm_->add(llvm::createInductiveRangeCheckEliminationPass()); // Pass
      break;
    case encodePassName("-indvars"): // Induction Variable Simplification
      pm_->add(llvm::createIndVarSimplifyPass()); // Pass
      break;
    case encodePassName("-licm"): // Loop Invariant Code Motion
      pm_->add(llvm::createLICMPass()); // Pass
      break;
    case encodePassName("-loop-sink"): // Loop sink
      pm_->add(llvm::createLoopSinkPass()); // Pass
      break;
    case encodePassName("-loop-predication"): // Loop prediction
      pm_->add(llvm::createLoopPredicationPass()); // Pass
      break;
    case encodePassName("-loop-interchange"): // Loop interchange
      pm_->add(llvm::createLoopInterchangePass()); // Pass
      break;
    case encodePassName("-loop-reduce"): // Loop Strength Reduction
      pm_->add(llvm::createLoopStrengthReducePass()); // Pass
      break;
    case encodePassName("-loop-unswitch"): // Unswitch loops
      pm_->add(llvm::createLoopUnswitchPass()); // Pass
      break;
    case encodePassName("-loop-instsimplify"): // Simplify instructions in loops
      pm_->add(llvm::createLoopInstSimplifyPass()); // Pass
      break;
    case encodePassName("-loop-unroll"): // Unroll loops
      pm_->add(llvm::createLoopUnrollPass()); // Pass
      break;
    case encodePassName("-loop-unroll-and-jam"): //  Unroll and Jam loops
      pm_->add(llvm::createLoopUnrollAndJamPass()); // Pass
      break;
    case encodePassName("-loop-reroll"): // Reroll loops
      pm_->add(llvm::createLoopRerollPass()); // Pass
      break;
    case encodePassName("-loop-rotate"): // Rotate Loops
      pm_->add(llvm::createLoopRotatePass()); // Pass
      break;
    case encodePassName("-loop-idiom"): // Recognize loop idioms
      pm_->add(llvm::createLoopIdiomPass()); // Pass
      break;
    case encodePassName("-loop-versioning-licm"): // Loop Versioning For LICM
      pm_->add(llvm::createLoopVersioningLICMPass()); // Pass
      break;
    case encodePassName("-reg2mem"): // Demote all values to stack slots
      pm_->add(llvm::createDemoteRegisterToMemoryPass()); // FunctionPass
      break;
    case encodePassName("-reassociate"): // Reassociate expressions
      pm_->add(llvm::createReassociatePass()); // FunctionPass
      break;
    case encodePassName("-jump-threading"): // Jump Threading
      pm_->add(llvm::createJumpThreadingPass()); // FunctionPass
      break;
    case encodePassName("-simplifycfg"): // Simplify the CFG
      pm_->add(llvm::createCFGSimplificationPass()); // FunctionPass
      break;
    case encodePassName("-flattencfg"): // Flatten the CFG
      pm_->add(llvm::createFlattenCFGPass()); // FunctionPass
      break;
    case encodePassName("-structurizecfg"): // Structurize the CFG
      pm_->add(llvm::createStructurizeCFGPass()); // Pass
      break;
    case encodePassName("-tailcallelim"): // Tail Call Elimination
      pm_->add(llvm::createTailCallEliminationPass()); // FunctionPass
      break;
    case encodePassName("-early-cse"): // Early CSE // FunctionPass
      pm_->add(llvm::createEarlyCSEPass()); // UseMemorySSA = false
      break;
    case encodePassName("-early-cse-memssa"): // Early CSE // FunctionPass
      pm_->add(llvm::createEarlyCSEPass(true)); // UseMemorySSA = true
      break;
    case encodePassName("-gvn-hoist"): // Early GVN Hoisting of Expressions
      pm_->add(llvm::createGVNHoistPass()); // FunctionPass
      break;
    case encodePassName("-gvn-sink"): // Early GVN sinking of Expressions
      pm_->add(llvm::createGVNSinkPass()); // FunctionPass
      break;
    case encodePassName("-mldst-motion"): // MergedLoadStoreMotion
      pm_->add(llvm::createMergedLoadStoreMotionPass()); // FunctionPass
      break;
    case encodePassName("-newgvn"): //  Global Value Numbering
      pm_->add(llvm::createNewGVNPass()); // FunctionPass
      break;
    case encodePassName("-div-rem-pairs"): // Hoist/decompose integer division and remainder
      pm_->add(llvm::createDivRemPairsPass()); // FunctionPass
      break;
    case encodePassName("-memcpyopt"): // MemCpy Optimization
      pm_->add(llvm::createMemCpyOptPass()); // FunctionPass
      break;
    case encodePassName("-loop-deletion"): // Delete dead loops
      pm_->add(llvm::createLoopDeletionPass()); // Pass
      break;
    case encodePassName("-consthoist"): // Constant Hoisting
      pm_->add(llvm::createConstantHoistingPass()); // FunctionPass
      break;
    case encodePassName("-sink"): // Code sinking
      pm_->add(llvm::createSinkingPass()); // FunctionPass
      break;
    case encodePassName("-loweratomic"): // Lower atomic intrinsics to non-atomic form
      pm_->add(llvm::createLowerAtomicPass()); // Pass
      break;
    case encodePassName("-lower-guard-intrinsic"): // Lower the guard intrinsic to normal control flow
      pm_->add(llvm::createLowerGuardIntrinsicPass()); // Pass
      break;
    case encodePassName("-lower-matrix-intrinsics"): // Lower the matrix intrinsics
      pm_->add(llvm::createLowerMatrixIntrinsicsPass()); // Pass
      break;
    case encodePassName("-lower-widenable-condition"): // Lower the widenable condition to default true value
      pm_->add(llvm::createLowerWidenableConditionPass()); // Pass
      break;
    case encodePassName("-mergeicmps"): // Merge contiguous icmps into a memcmp
      pm_->add(llvm::createMergeICmpsLegacyPass()); // Pass
      break;
    case encodePassName("-correlated-propagation"): // Value Propagation
      pm_->add(llvm::createCorrelatedValuePropagationPass()); // Pass
      break;
    case encodePassName("-infer-address-spaces"): // Infer address spaces
      pm_->add(llvm::createInferAddressSpacesPass()); // FunctionPass
      break;
    case encodePassName("-lower-expect"): // Lower 'expect' Intrinsics
      pm_->add(llvm::createLowerExpectIntrinsicPass()); // FunctionPass
      break;
    case encodePassName("-lower-constant-intrinsics"): //  Lower constant intrinsics
      pm_->add(llvm::createLowerConstantIntrinsicsPass()); // FunctionPass
      break;
    case encodePassName("-partially-inline-libcalls"): //Partially inline calls to library functions
      pm_->add(llvm::createPartiallyInlineLibCallsPass()); // FunctionPass
      break;
    case encodePassName("-separate-const-offset-from-gep"): // Split GEPs to a variadic base and a constant offset for better CSE
      pm_->add(llvm::createSeparateConstOffsetFromGEPPass()); // FunctionPass
      break;
    case encodePassName("-speculative-execution"): // Speculative execution
      pm_->add(llvm::createSpeculativeExecutionPass()); // FunctionPass
      break;
    case encodePassName("-slsr"): // Straight line strength reduction
      pm_->add(llvm::createStraightLineStrengthReducePass());// FunctionPass
      break;
    case encodePassName("-place-safepoints"): // Place Safepoints
      pm_->add(llvm::createPlaceSafepointsPass());// FunctionPass
      break;
    case encodePassName("-rewrite-statepoints-for-gc"): // Make relocations explicit at statepoints
      pm_->add(llvm::createRewriteStatepointsForGCLegacyPass());// ModulePass
      break;
    case encodePassName("-float2int"): // Float to int
      pm_->add(llvm::createFloat2IntPass()); // FunctionPass
      break;
    case encodePassName("-nary-reassociate"): // Nary reassociation
      pm_->add(llvm::createNaryReassociatePass()); // FunctionPass
      break;
    case encodePassName("-loop-distribute"): // Loop Distribution
      pm_->add(llvm::createLoopDistributePass()); // FunctionPass
      break;
    case encodePassName("-loop-fusion"): // Loop Fusion
      pm_->add(llvm::createLoopFusePass()); // FunctionPass
      break;
    case encodePassName("-loop-load-elim"): // Loop Load Elimination
      pm_->add(llvm::createLoopLoadEliminationPass()); // FunctionPass
      break;
    case encodePassName("-loop-versioning"): // Loop Versioning
      pm_->add(llvm::createLoopVersioningPass()); // FunctionPass
      break;
    case encodePassName("-loop-data-prefetch"): // Loop Data Prefetch
      pm_->add(llvm::createLoopDataPrefetchPass()); // FunctionPass
      break;
    case encodePassName("-name-anon-globals"): // Provide a name to nameless globals
      pm_->add(llvm::createNameAnonGlobalPass()); // ModulePass
      break;
    case encodePassName("-canonicalize-aliases"): // Canonicalize aliases
      pm_->add(llvm::createCanonicalizeAliasesPass()); // ModulePass
      break;
    case encodePassName("-libcalls-shrinkwrap"): // Conditionally eliminate dead library calls
      pm_->add(llvm::createLibCallsShrinkWrapPass()); // FunctionPass
      break;
    case encodePassName("-loop-simplifycfg"): // Simplify loop CFG
      pm_->add(llvm::createLoopSimplifyCFGPass()); // Pass
      break;

    // "Transforms/Scalar/Scalarizer.h"
    case encodePassName("-scalarizer"): // Scalarize vector operations
      pm_->add(llvm::createScalarizerPass()); // FunctionPass
      break;

    // "Transforms/Scalar/GVN.h"
    case encodePassName("-gvn"): // Global Value Numbering
      pm_->add(llvm::createGVNPass()); // FunctionPass
      break;

    // "Transforms/Scalar/InstSimplifyPass.h"
    case encodePassName("-instsimplify"): // Remove redundant instructions
      pm_->add(llvm::createInstSimplifyLegacyPass()); // FunctionPass
      break;

    // "Transforms/InstCombine/InstCombine.h"
    case encodePassName("-instcombine"): // Combine redundant instructions
      pm_->add(llvm::createInstructionCombiningPass()); // FunctionPass
      break;

    // "Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
    case encodePassName("-aggressive-instcombine"): // Combine pattern based expressions
      pm_->add(llvm::createAggressiveInstCombinerPass()); // FunctionPass
      break;

    // "Transforms/Utils.h"
    case encodePassName("-metarenamer"): // Assign new names to everything
      pm_->add(llvm::createMetaRenamerPass()); // ModulePass
      break;
    case encodePassName("-lowerinvoke"): // Lower invoke and unwind, for unwindless code generators
      pm_->add(llvm::createLowerInvokePass()); // FunctionPass
      break;
    case encodePassName("-instnamer"): // Assign names to anonymous instructions
      pm_->add(llvm::createInstructionNamerPass()); // FunctionPass
      break;
    case encodePassName("-lowerswitch"): // Lower SwitchInst's to branches
      pm_->add(llvm::createLowerSwitchPass()); // FunctionPass
      break;
    case encodePassName("-ee-instrument"): //Instrument function entry/exit with calls to e.g. mcount() (pre inlining)
      pm_->add(llvm::createEntryExitInstrumenterPass()); // FunctionPass
      break;
    case encodePassName("-post-inline-ee-instrument"): //Instrument function entry/exit with calls to e.g. mcount() (post inlining)
      pm_->add(llvm::createPostInlineEntryExitInstrumenterPass()); // FunctionPass
      break;
    case encodePassName("-break-crit-edges"): // Break critical edges in CFG
      pm_->add(llvm::createBreakCriticalEdgesPass()); // FunctionPass
      break;
    case encodePassName("-lcssa"): // Loop-Closed SSA Form Pass
      pm_->add(llvm::createLCSSAPass()); // Pass
      break;
    case encodePassName("-mem2reg"): // Promote Memory to Register
      pm_->add(llvm::createPromoteMemoryToRegisterPass()); // FunctionPass
      break;
    case encodePassName("-loop-simplify"): // Simplify loop CFG
      pm_->add(llvm::createLoopSimplifyPass()); // Pass
      break;
    case encodePassName("-strip-nonlinetable-debuginfo"): // Strip all debug info except linetables
      pm_->add(llvm::createStripNonLineTableDebugInfoPass()); // ModulePass
      break;
    case encodePassName("-chr"): // Reduce control height in the hot paths
      pm_->add(llvm::createControlHeightReductionLegacyPass()); // FunctionPass
      break;
    case encodePassName("-inject-tli-mappings"): //Inject TLI Mappings
      pm_->add(llvm::createInjectTLIMappingsLegacyPass()); // FunctionPass
      break;

    // "Transforms/Vectorize.h"
    case encodePassName("-loop-vectorize"): // Loop Vectorization
      pm_->add(llvm::createLoopVectorizePass()); // Pass
      break;
    case encodePassName("-slp-vectorizer"): // SLP Vectorizer
      pm_->add(llvm::createSLPVectorizerPass()); // Pass
      break;
    case encodePassName("-load-store-vectorizer"): // Vectorize load and store instructions
      pm_->add(llvm::createLoadStoreVectorizerPass()); // Pass
      break;

    //Transforms/IPO/InferFunctionAttrs.h
    case encodePassName("-inferattrs"): // Infer set function attributes
      pm_->add(llvm::createInferFunctionAttrsLegacyPass()); // Pass
      break;

    // Transforms/IPO/ForceFunctionAttrs.h
    case encodePassName("-forceattrs"): // Force set function attributes
      pm_->add(llvm::createForceFunctionAttrsLegacyPass()); // Pass'
      break;

    // "Transforms/IPO.h"
    case encodePassName("-strip"): // Strip all symbols from a module
      pm_->add(llvm::createStripSymbolsPass()); // ModulePass
      break;
    case encodePassName("-strip-nondebug"): // Strip all symbols, except dbg symbols, from a module
      pm_->add(llvm::createStripNonDebugSymbolsPass()); // ModulePass
      break;
    case encodePassName("-strip-debug-declare"): // Strip all llvm.dbg.declare intrinsics
      pm_->add(llvm::createStripDebugDeclarePass()); // ModulePass
      break;
    case encodePassName("-strip-dead-debug-info"): // Strip debug info for unused symbols
      pm_->add(llvm::createStripDeadDebugInfoPass()); // ModulePass
      break;
    case encodePassName("-constmerge"): // Merge Duplicate Global Constants
      pm_->add(llvm::createConstantMergePass()); // ModulePass
      break;
    case encodePassName("-globalopt"): // Global Variable Optimizer
      pm_->add(llvm::createGlobalOptimizerPass()); // ModulePass
      break;
    case encodePassName("-globaldce"): // Dead Global Elimination
      pm_->add(llvm::createGlobalDCEPass()); // ModulePass
      break;
    case encodePassName("-elim-avail-extern"): // Eliminate Available Externally Globals
      pm_->add(llvm::createEliminateAvailableExternallyPass()); // ModulePass
      break;
    case encodePassName("-function-import"): // Summary Based Function Import
      pm_->add(llvm::createFunctionImportPass()); // Pass
      break;
    case encodePassName("-inline"): // Function Integration/Inlining
      pm_->add(llvm::createFunctionInliningPass()); // Pass
      break;
    case encodePassName("-prune-eh"): // Remove unused exception handling info
      pm_->add(llvm::createPruneEHPass()); // Pass
      break;
    case encodePassName("-internalize"): // Internalize Global Symbols
      pm_->add(llvm::createInternalizePass()); // ModulePass
      break;
    case encodePassName("-deadargelim"): // Dead Argument Elimination
      pm_->add(llvm::createDeadArgEliminationPass()); // ModulePass
      break;
    case encodePassName("-argpromotion"): // Promote 'by reference' arguments to scalars
      pm_->add(llvm::createArgumentPromotionPass()); // Pass
      break;
    case encodePassName("-ipconstprop"): // Interprocedural constant propagation
      pm_->add(llvm::createIPConstantPropagationPass()); // ModulePass
      break;
    case encodePassName("-ipsccp"): //Interprocedural Sparse Conditional Constant Propagation
      pm_->add(llvm::createIPSCCPPass()); // ModulePass
      break;
    case encodePassName("-loop-extract"): // Extract loops into new functions
      pm_->add(llvm::createLoopExtractorPass()); // Pass
      break;
    case encodePassName("-loop-extract-single"): // Extract at most one loop into a new function
      pm_->add(llvm::createSingleLoopExtractorPass()); // Pass
      break;
    case encodePassName("-extract-blocks"): // Extract basic blocks from module
      pm_->add(llvm::createBlockExtractorPass()); // ModulePass
      break;
    case encodePassName("-strip-dead-prototypes"): // Strip Unused Function Prototypes
      pm_->add(llvm::createStripDeadPrototypesPass()); // ModulePass
      break;
    case encodePassName("-mergefunc"): // Merge Functions
      pm_->add(llvm::createMergeFunctionsPass()); // ModulePass
      break;
    case encodePassName("-hotcoldsplit"): // Hot Cold Splitting
      pm_->add(llvm::createHotColdSplittingPass()); // ModulePass
      break;
    case encodePassName("-partial-inliner"): // Partial Inliner
      pm_->add(llvm::createPartialInliningPass()); // ModulePass
      break;
    case encodePassName("-barrier"): //  A No-Op Barrier Pass
      pm_->add(llvm::createBarrierNoopPass());// ModulePass
      break;
    case encodePassName("-called-value-propagation"): // Called Value Propagation
      pm_->add(llvm::createCalledValuePropagationPass()); // ModulePass
      break;
    case encodePassName("-cross-dso-cfi"): // Cross-DSO CF
      pm_->add(llvm::createCrossDSOCFIPass()); // ModulePass
      break;
    case encodePassName("-globalsplit"): // Global splitter
      pm_->add(llvm::createGlobalSplitPass()); // ModulePass
      break;

    // "Analysis/Passes.h"
    case encodePassName("-pa-eval"): // Evaluate ProvenanceAnalysis on all pairs
      pm_->add(llvm::createPAEvalPass()); // FunctionPass
      break;
    case encodePassName("-lazy-value-info"): // Lazy Value Information Analysis
      pm_->add(llvm::createLazyValueInfoPass()); // FunctionPass
      break;
    case encodePassName("-da"): // Dependence Analysis
      pm_->add(llvm::createDependenceAnalysisWrapperPass()); // FunctionPass
      break;
    case encodePassName("-cost-model"): // Cost Model Analysis
      pm_->add(llvm::createCostModelAnalysisPass()); // FunctionPass
      break;
    case encodePassName("-delinearize"): // Delinearization
      pm_->add(llvm::createDelinearizationPass()); // FunctionPass
      break;
    case encodePassName("-divergence"): // Divergence Analysis
      pm_->add(llvm::createLegacyDivergenceAnalysisPass()); // FunctionPass
      break;
    case encodePassName("-instcount"): // Counts the various types of Instructions
      pm_->add(llvm::createInstCountPass()); // FunctionPass
      break;
    case encodePassName("-regions"): // Detect single entry single exit regions
      pm_->add(llvm::createRegionInfoPass()); // FunctionPass
      break;

    // Analysis/BasicAliasAnalysis.h
    case encodePassName("-basicaa"): // Basic Alias Analysis (stateless AA impl)
      pm_->add(llvm::createBasicAAWrapperPass()); // FunctionPass
      break;

    // Analysis/AliasAnalysis.h
    case encodePassName("-aa"): // Function Alias Analysis Results
      pm_->add(llvm::createAAResultsWrapperPass()); // FunctionPass
      break;

    // Analysis/GlobalsModRef.h
    case encodePassName("-globals-aa"): // Globals Alias Analysis
      pm_->add(llvm::createGlobalsAAWrapperPass()); // ModulePass
      break;

    // Analysis/TypeBasedAliasAnalysis.h
    case encodePassName("-tbaa"): // Type-Based Alias Analysis
      pm_->add(llvm::createTypeBasedAAWrapperPass()); // ImmutablePass
      break;

    // "CodeGen/Passes.h"
    case encodePassName("-codegenprepare"): // Optimize for code generation
      pm_->add(llvm::createCodeGenPreparePass()); // FunctionPass
      break;

    default:
      std::string msg = "Trying to use an invalid optimization pass (";
      msg += pass;
      msg += ")!";
      throw std::runtime_error(msg);
      break;
    }
}

void LLVMDriver::Invoke(std::string filename, std::vector<::llvm::Pass *> passes) {
  if (!passes.empty()) {
      // Parse the input LLVM IR file into a module.
      SMDiagnostic err;
      LLVMContext context;
	     // src == filename
      std::unique_ptr<::llvm::Module> Module(parseIRFile(filename, err, context));

      // src == read(filename)
      //MemoryBufferRef mb = MemoryBuffer::getMemBuffer(src)->getMemBufferRef();
      //std::unique_ptr<::llvm::Module> module = parseIR(mb, err, context);

      if (!Module) {
        err.print(filename.c_str(), errs());
        throw std::runtime_error("Failed parse IR file");
      }

      ::llvm::remove_fatal_error_handler();

      // Register other llvm passes.
      PassRegistry &reg = *PassRegistry::getPassRegistry();
      initializeCallGraphWrapperPassPass(reg);
      initializeMemorySSAWrapperPassPass(reg);

      /*
      // OLD VERSION
      initializeStripSymbolsPass(reg);
      // Setup the pass manager and add passes.
      pm_.reset(new legacy::PassManager());
      for (auto pass : passes) {
          pm_->add(pass);
      }

      // Run passes.
      pm_->run(*Module);
      */

      // Setup the pass manager.
      pm_.reset(new legacy::PassManager());

      if (!optimizations_.empty()){
        // Add Analysis and transformations passes
        addOptimizationPasses();
      }

      // Add passes.
      for (auto pass : passes) {
          pm_->add(pass);
      }

      // Run passes.
      pm_->run(*Module);
  }

}

}  // namespace compy
