/*
 This file is distributed under the University of Illinois Open Source
 License. See LICENSE for details.
*/

#pragma once

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/LoopInfo.h"

#include <map>
#include <set>

namespace compy {
namespace llvm {
namespace wlcost {

  class BranchPredictionInfo {
  public:
    typedef std::pair<const ::llvm::BasicBlock *, const ::llvm::BasicBlock *> Edge;
  private:
    std::set<Edge> listBackEdges_, listExitEdges_;
    std::map<const ::llvm::BasicBlock *, unsigned> backEdgesCount_;
    std::set<const ::llvm::BasicBlock *> listCalls_, listStores_;

    ::llvm::DominatorTree *dominatorTree_;
    ::llvm::PostDominatorTree *postDominatorTree_;
    ::llvm::LoopInfo *loopInfo_;

    void findBackAndExitEdges(::llvm::Function &F);
    void findCallsAndStores(::llvm::Function &F);
  public:
    explicit BranchPredictionInfo(::llvm::DominatorTree *DT, ::llvm::LoopInfo *LI,
                                  ::llvm::PostDominatorTree *PDT = NULL);

    void buildInfo(::llvm::Function &F);
    unsigned countBackEdges(::llvm::BasicBlock *BB) const;
    bool callsExit(::llvm::BasicBlock *BB) const;
    bool isBackEdge(const Edge &edge) const;
    bool isExitEdge(const Edge &edge) const;
    bool hasCall(const ::llvm::BasicBlock *BB) const;
    bool hasStore(const ::llvm::BasicBlock *BB) const;

    inline ::llvm::DominatorTree *getDominatorTree() const { return dominatorTree_; }
    inline ::llvm::PostDominatorTree *getPostDominatorTree() const { return postDominatorTree_; }
    inline ::llvm::LoopInfo *getLoopInfo() const { return loopInfo_; }
  };

}  // namespace cost
}  // namespace llvm
}  // namespace compy
