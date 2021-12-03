/*
 This file is distributed under the University of Illinois Open Source
 License. See LICENSE for details.
*/

#include "branch_prediction_info.h"

using namespace ::llvm;

namespace compy {
namespace llvm {
namespace wlcost {

BranchPredictionInfo::BranchPredictionInfo(DominatorTree *DT,
					   LoopInfo *LI,
                                           PostDominatorTree *PDT) {
  this->dominatorTree_ = DT;
  this->postDominatorTree_ = PDT;
  this->loopInfo_ = LI;
}

/// FindBackAndExitEdges - Search for back and exit edges for all blocks
/// within the function loops, calculated using loop information.
void BranchPredictionInfo::findBackAndExitEdges(Function &F) {
  std::set<const BasicBlock *> LoopsVisited;
  std::set<const BasicBlock *> BlocksVisited;

  for (LoopInfo::iterator LIT = loopInfo_->begin(), LIE = loopInfo_->end(); LIT != LIE; ++LIT) {
    Loop *rootLoop = *LIT;
    BasicBlock *rootHeader = rootLoop->getHeader();

    // Check if we already visited this loop.
    if (LoopsVisited.count(rootHeader))
      continue;

    // Create a stack to hold loops (inner most on the top).
    SmallVector<Loop *, 8> Stack;
    SmallPtrSet<const BasicBlock *, 8> InStack;

    // Put the current loop into the Stack.
    Stack.push_back(rootLoop);
    InStack.insert(rootHeader);

    do {
      Loop *loop = Stack.back();

      // Search for new inner loops.
      bool foundNew = false;
      for (Loop::iterator I = loop->begin(), E = loop->end(); I != E; ++I) {
        Loop *innerLoop = *I;
        BasicBlock *innerHeader = innerLoop->getHeader();

        // Skip visited inner loops.
        if (!LoopsVisited.count(innerHeader)) {
          Stack.push_back(innerLoop);
          InStack.insert(innerHeader);
          foundNew = true;
          break;
        }
      }

      // If a new loop is found, continue.
      // Otherwise, it is time to expand it, because it is the most inner loop
      // yet unprocessed.
      if (foundNew)
        continue;

      // The variable "loop" is now the unvisited inner most loop.
      BasicBlock *header = loop->getHeader();

      // Search for all basic blocks on the loop.
      for (Loop::block_iterator LBI = loop->block_begin(),
           LBE = loop->block_end(); LBI != LBE; ++LBI) {
        BasicBlock *lpBB = *LBI;
        if (!BlocksVisited.insert(lpBB).second)
          continue;

        // Set the number of back edges to this loop head (lpBB) as zero.
        backEdgesCount_[lpBB] = 0;

        // For each loop block successor, check if the block pointing is
        // outside the loop.
        Instruction *TI = lpBB->getTerminator();
        for (unsigned s = 0; s < TI->getNumSuccessors(); ++s) {
          BasicBlock *successor = TI->getSuccessor(s);
          Edge edge = std::make_pair(lpBB, successor);

          // If the successor matches any loop header on the stack,
          // then it is a backedge.
          if (InStack.count(successor)) {
            listBackEdges_.insert(edge);
            ++backEdgesCount_[lpBB];
          }

          // If the successor is not present in the loop block list, then it is
          // an exit edge.
          if (!loop->contains(successor))
            listExitEdges_.insert(edge);
        }
      }

      // Cleaning the visited loop.
      LoopsVisited.insert(header);
      Stack.pop_back();
      InStack.erase(header);
    } while (!InStack.empty());
  }
}

/// FindCallsAndStores - Search for call and store instruction on basic blocks.
void BranchPredictionInfo::findCallsAndStores(Function &F) {
  // Run through all basic blocks of functions.
  for (const auto &BB : F.getBasicBlockList()) {
    // We only need to know if a basic block contains ONE call and/or ONE store.
    bool calls = false;
    bool stores = false;

    // If the terminator instruction is an InvokeInstruction, add it directly.
    // An invoke instruction can be interpreted as a call.
    if (isa<InvokeInst>(BB.getTerminator())) {
      listCalls_.insert(&BB);
      calls = true;
    }

    // Run over through all basic block searching for any calls.
    //for (BasicBlock::iterator BI = BB.begin(), BE = BB.end(); BI != BE; ++BI) {
    for (const auto &I : BB) {
      // If we found one of each, we don't need to search anymore.
      if (stores && calls)
        break;

      // If we haven't found a store yet, test the instruction
      // and mark it if it is a store instruction.
      if (!stores && isa<StoreInst>(I)) {
        listStores_.insert(&BB);
        stores = true;
      }

      // If we haven't found a call yet, test the instruction
      // and mark it if it is a call instruction.
      if (!calls && isa<CallInst>(I)) {
        listCalls_.insert(&BB);
        calls = true;
      }
    }
  }
}

/// BuildInfo - Build the list of back edges, exit edges, calls and stores.
void BranchPredictionInfo::buildInfo(Function &F) {
  backEdgesCount_.clear();
  listBackEdges_.clear();
  listExitEdges_.clear();
  listCalls_.clear();
  listStores_.clear();

  // Find the list of back edges and exit edges for all of the edges in the
  // respective function and build a list.
  findBackAndExitEdges(F);

  // Find all the basic blocks in the function "F" that contains calls or
  // stores and build a list.
  findCallsAndStores(F);
}

/// CountBackEdges - Given a basic block, count the number of successor
/// that are back edges.
unsigned BranchPredictionInfo::countBackEdges(BasicBlock *BB) const {
  std::map<const BasicBlock *, unsigned>::const_iterator it =
      backEdgesCount_.find(BB);
  return it != backEdgesCount_.end() ? it->second : 0;
}

/// CallsExit - Check whenever a basic block contains a call to exit.
bool BranchPredictionInfo::callsExit(BasicBlock *BB) const {
  Instruction *TI = BB->getTerminator();
  return (isa<ResumeInst>(TI));
}

/// isBackEdge - Verify if an edge is a back edge.
bool BranchPredictionInfo::isBackEdge(const Edge &edge) const {
  return listBackEdges_.count(edge);
}

/// isExitEdge - Verify if an edge is an exit edge.
bool BranchPredictionInfo::isExitEdge(const Edge &edge) const {
  return listExitEdges_.count(edge);
}

/// hasCall - Verify if a basic block contains a call.
bool BranchPredictionInfo::hasCall(const BasicBlock *BB) const {
  return listCalls_.count(BB);
}

/// hasStore - Verify if any instruction of a basic block is a store.
bool BranchPredictionInfo::hasStore(const BasicBlock *BB) const {
  return listStores_.count(BB);
}

}  // namespace cost
}  // namespace llvm
}  // namespace compy
