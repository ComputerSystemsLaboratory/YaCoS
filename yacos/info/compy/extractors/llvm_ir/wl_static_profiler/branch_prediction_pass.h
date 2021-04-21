/*
 This file is distributed under the University of Illinois Open Source
 License. See LICENSE for details.
*/

#pragma once


#include "llvm/Pass.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include "branch_heuristics_info.h"

#include <map>

namespace compy {
namespace llvm {
namespace wlcost {

  class BranchPredictionPass : public ::llvm::FunctionPass {
  public:
    typedef std::pair<const ::llvm::BasicBlock *, const ::llvm::BasicBlock *> Edge;
  private:
    BranchPredictionInfo *branchPredictionInfo_;
    BranchHeuristicsInfo *branchHeuristicsInfo_;

    std::map<Edge, double> edgeProbabilities_;

    void calculateBranchProbabilities(::llvm::BasicBlock *BB);
    void addEdgeProbability(BranchHeuristics heuristic,
			    const ::llvm::BasicBlock *root,
                            Prediction pred);
  public:

    static char ID;

    BranchPredictionPass() : ::llvm::FunctionPass(ID), branchPredictionInfo_(nullptr), branchHeuristicsInfo_(nullptr) {}
    ~BranchPredictionPass() { Clear(); }

    bool runOnFunction(::llvm::Function &func) override;
    void getAnalysisUsage(::llvm::AnalysisUsage &au) const override;

    double getEdgeProbability(const ::llvm::BasicBlock *src, const ::llvm::BasicBlock *dst) const;
    double getEdgeProbability(Edge &edge) const;
    const BranchPredictionInfo *getInfo() const;
    void Clear();

  };

}  // namespace cost
}  // namespace llvm
}  // namespace compy
