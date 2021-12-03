/*
 This file is distributed under the University of Illinois Open Source
 License. See LICENSE for details.
*/

#pragma once

#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Dominators.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Support/Debug.h"

#include "branch_prediction_info.h"

#include <utility>

namespace compy {
namespace llvm {
namespace wlcost {

  class BranchPredictionInfo;

  // All possible branch heuristics.
  enum BranchHeuristics {
    LOOP_BRANCH_HEURISTIC = 0,
    POINTER_HEURISTIC,
    CALL_HEURISTIC,
    OPCODE_HEURISTIC,
    LOOP_EXIT_HEURISTIC,
    RETURN_HEURISTIC,
    STORE_HEURISTIC,
    LOOP_HEADER_HEURISTIC,
    GUARD_HEURISTIC
  };

  struct BranchProbabilities {
    enum BranchHeuristics heuristic;
    const float probabilityTaken; // probability of taken a branch
    const float probabilityNotTaken; // probability of not taken a branch
    const char *name;
  };

  typedef std::pair<const ::llvm::BasicBlock *, const ::llvm::BasicBlock *> Prediction;

  class BranchHeuristicsInfo {
  private:
    BranchPredictionInfo *branchPredictionInfo_;
    ::llvm::PostDominatorTree *postDominatorTree_;
    ::llvm::LoopInfo *loopInfo_;

    Prediction empty;

    static const unsigned numBranchHeuristics_ = 9;
    static const struct BranchProbabilities probList[numBranchHeuristics_];

    Prediction matchLoopBranchHeuristic(::llvm::BasicBlock *root) const;
    Prediction matchPointerHeuristic(::llvm::BasicBlock *root) const;
    Prediction matchCallHeuristic(::llvm::BasicBlock *root) const;
    Prediction matchOpcodeHeuristic(::llvm::BasicBlock *root) const;
    Prediction matchLoopExitHeuristic(::llvm::BasicBlock *root) const;
    Prediction matchReturnHeuristic(::llvm::BasicBlock *root) const;
    Prediction matchStoreHeuristic(::llvm::BasicBlock *root) const;
    Prediction matchLoopHeaderHeuristic(::llvm::BasicBlock *root) const;
    Prediction matchGuardHeuristic(::llvm::BasicBlock *root) const;
  public:

    typedef std::pair<const ::llvm::BasicBlock *, const ::llvm::BasicBlock *> Edge;

    explicit BranchHeuristicsInfo(BranchPredictionInfo *BPI);

    Prediction matchHeuristic(BranchHeuristics bh, ::llvm::BasicBlock *root) const;

    inline static unsigned getNumHeuristics() { return numBranchHeuristics_; }
    inline static enum BranchHeuristics getHeuristic(unsigned idx) { return probList[idx].heuristic; }
    inline static float getProbabilityTaken(enum BranchHeuristics bh) { return probList[bh].probabilityTaken; }
    inline static float getProbabilityNotTaken(enum BranchHeuristics bh) { return probList[bh].probabilityNotTaken; }
    inline static const char *getHeuristicName(enum BranchHeuristics bh) { return probList[bh].name; }
  };

}  // namespace cost
}  // namespace llvm
}  // namespace compy
