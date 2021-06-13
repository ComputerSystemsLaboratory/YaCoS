/*
 This file is distributed under the University of Illinois Open Source
 License. See LICENSE for details.
*/

#pragma once

#include "llvm/Pass.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>
#include <algorithm>
#include <map>
#include <set>

namespace compy {
namespace llvm {
namespace wlcost {

  class BranchPredictionPass;

  class BlockEdgeFrequencyPass : public ::llvm::FunctionPass {
  public:
    typedef std::pair<const ::llvm::BasicBlock *, const ::llvm::BasicBlock *> Edge;
  private:
    static const double epsilon_;

    ::llvm::LoopInfo *loopInfo_;
    BranchPredictionPass *branchPredictionPass_;

    std::set<const ::llvm::BasicBlock *> notVisited_;
    std::set<const ::llvm::Loop *> loopsVisited_;
    std::map<Edge, double> backEdgeProbabilities_;
    std::map<Edge, double> edgeFrequencies_;
    std::map<const ::llvm::BasicBlock *, double> blockFrequencies_;

    void markReachable(::llvm::BasicBlock *root);
    void propagateLoop(const ::llvm::Loop *loop);
    void propagateFreq(::llvm::BasicBlock *head);

  public:
    static char ID;

    BlockEdgeFrequencyPass() : ::llvm::FunctionPass(ID) {}

    void getAnalysisUsage(::llvm::AnalysisUsage &au) const override;
    bool runOnFunction(::llvm::Function &func) override;

    double getEdgeFrequency(const ::llvm::BasicBlock *src, const ::llvm::BasicBlock *dst) const;
    double getEdgeFrequency(Edge &edge) const;
    double getBlockFrequency(const ::llvm::BasicBlock *BB) const;
    double getBackEdgeProbabilities(Edge &edge);

  };

}  // namespace graph
}  // namespace llvm
}  // namespace compy
