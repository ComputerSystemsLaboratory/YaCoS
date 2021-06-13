/*
 This file is distributed under the University of Illinois Open Source
 License. See LICENSE for details.
*/

#include "branch_prediction_pass.h"
#include "branch_prediction_info.h"
#include "branch_heuristics_info.h"

using namespace ::llvm;

namespace compy {
namespace llvm {
namespace wlcost {

void BranchPredictionPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<DominatorTreeWrapperPass>();
  au.addRequired<PostDominatorTreeWrapperPass>();
  au.addRequired<LoopInfoWrapperPass>();
  au.setPreservesAll();
}

bool BranchPredictionPass::runOnFunction(Function &func) {
  // To perform the branch prediction, the following passes are required.
  DominatorTree *DT =  &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  PostDominatorTree *PDT = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
  LoopInfo *LI =  &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  // Clear previously calculated data.
  Clear();

  // Build all required information to run the branch prediction pass.
  branchPredictionInfo_ = new BranchPredictionInfo(DT, LI, PDT);
  branchPredictionInfo_->buildInfo(func);

  // Create the class to check branch heuristics.
  branchHeuristicsInfo_ = new BranchHeuristicsInfo(branchPredictionInfo_);

  // Run over all basic blocks of a function calculating branch probabilities.
  //for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
  for (auto &bb : func.getBasicBlockList())
    // Calculate the likelihood of the successors of this basic block.
    calculateBranchProbabilities(&bb);

  // Delete unnecessary branch heuristic info.
  delete branchHeuristicsInfo_;
  branchHeuristicsInfo_ = NULL;

  return false;
}

/// getEdgeProbability - Find the edge probability based on the source and
/// the destination basic block.  If the edge is not found, return 1.0
/// (probability of 100% of being taken).
double BranchPredictionPass::getEdgeProbability(const BasicBlock *src,
                                                const BasicBlock *dst) const {
  // Create the edge.
  Edge edge = std::make_pair(src, dst);

  // Find the profile based on the edge.
  return getEdgeProbability(edge);
}

/// getEdgeProbability - Find the edge probability. If the edge is not found,
/// return 1.0 (probability of 100% of being taken).
double BranchPredictionPass::getEdgeProbability(Edge &edge) const {
  // Search for the edge on the list.
  std::map<Edge, double>::const_iterator I = edgeProbabilities_.find(edge);

  // If edge was found, return it. Otherwise return the default value,
  // meaning that there is no profile known for this edge. The default value
  // is 1.0, meaning that the branch is taken with 100% likelihood.
  return I != edgeProbabilities_.end() ? I->second : 1.0;
}

/// getInfo - Get branch prediction information regarding edges and blocks.
const BranchPredictionInfo *BranchPredictionPass::getInfo() const {
  return branchPredictionInfo_;
}

/// Clear - Empty all stored information.
void BranchPredictionPass::Clear() {
  // Clear edge probabilities.
  edgeProbabilities_.clear();

  // Free previously calculated branch prediction info class.
  if (branchPredictionInfo_) {
    delete branchPredictionInfo_;
    branchPredictionInfo_ = NULL;
  }

  // Free previously calculated branch heuristics class.
  if (branchHeuristicsInfo_) {
    delete branchHeuristicsInfo_;
    branchHeuristicsInfo_ = NULL;
  }
}

/// CalculateBranchProbabilities - Implementation of the algorithm proposed
/// by Wu (1994) to calculate the probabilities of all the successors of a
/// basic block.
void BranchPredictionPass::calculateBranchProbabilities(BasicBlock *BB) {
  // Obtain the last instruction.
  Instruction *TI = BB->getTerminator();

  // Find the total number of successors (variable "m" in Wu's paper)
  unsigned successors = TI->getNumSuccessors();

  // Find the total number of back edges (variable "n" in Wu's paper)
  unsigned backedges = branchPredictionInfo_->countBackEdges(BB);

  // The basic block must have successors,
  // so that we can have something to profile
  if (successors != 0) {
    // If a block calls exit, then assume that every successor of this
    // basic block is never going to be reached.
    if (branchPredictionInfo_->callsExit(BB)) {
      // According to the paper, successors that contains an exit call have a
      // probability of 0% to be taken.
      for (unsigned s = 0; s < successors; ++s) {
        BasicBlock *succ = TI->getSuccessor(s);
        Edge edge = std::make_pair(BB, succ);
        edgeProbabilities_[edge] = 0.0f;
      }
    } else if (backedges > 0 && backedges < successors) {
      // Has some back edges, but not all.
      for (unsigned s = 0; s < successors; ++s) {
        BasicBlock *succ = TI->getSuccessor(s);
        Edge edge = std::make_pair(BB, succ);

        // Check if edge is a backedge.
        if (branchPredictionInfo_->isBackEdge(edge)) {
          edgeProbabilities_[edge] =
              branchHeuristicsInfo_->getProbabilityTaken(LOOP_BRANCH_HEURISTIC) / backedges;
        } else {
          // The other edge, the one that is not a back edge, is in most cases
          // an exit edge. However, there are situations in which this edge is
          // an exit edge of an inner loop, but not for the outer loop. So,
          // consider the other edges always as an exit edge.

          edgeProbabilities_[edge] =
              branchHeuristicsInfo_->getProbabilityNotTaken(LOOP_BRANCH_HEURISTIC) /
              (successors - backedges);
        }
      }
    } else if (backedges > 0 || successors != 2) {
      // This part handles the situation involving switch statements.
      // Every switch case has a equal likelihood to be taken.
      // Calculates the probability given the total amount of cases clauses.
      for (unsigned s = 0; s < successors; ++s) {
        BasicBlock *succ = TI->getSuccessor(s);
        Edge edge = std::make_pair(BB, succ);
        edgeProbabilities_[edge] = 1.0f / successors;
      }
    } else {
      // Here we can only handle basic blocks with two successors (branches).
      // This assertion might never occur due to conditions meet above.
      assert(successors == 2 && "Expected a two way branch");

      // Identify the two branch edges.
      Edge trueEdge = std::make_pair(BB, TI->getSuccessor(0));
      Edge falseEdge = std::make_pair(BB, TI->getSuccessor(1));

      // Initial branch probability. If no heuristic matches, than each edge
      // has a likelihood of 50% to be taken.
      edgeProbabilities_[trueEdge] = 0.5f;
      edgeProbabilities_[falseEdge] = 0.5f;

      // Run over all heuristics implemented in BranchHeuristics class.
      for (unsigned h = 0; h < branchHeuristicsInfo_->getNumHeuristics(); ++h) {
        // Retrieve the next heuristic.
        BranchHeuristics heuristic = branchHeuristicsInfo_->getHeuristic(h);

        // If the heuristic matched, add the edge probability to it.
        Prediction pred = branchHeuristicsInfo_->matchHeuristic(heuristic, BB);

        // Heuristic matched.
        if (pred.first)
          // Recalculate edge probability.
          addEdgeProbability(heuristic, BB, pred);
      }

    }
  }
}

/// addEdgeProbability - If a heuristic matches, calculates the edge probability
/// combining previous predictions acquired.
void BranchPredictionPass::addEdgeProbability(BranchHeuristics heuristic,
                                              const BasicBlock *root,
                                              Prediction pred) {
  const BasicBlock *successorTaken = pred.first;
  const BasicBlock *successorNotTaken = pred.second;

  // Get the edges.
  Edge edgeTaken = std::make_pair(root, successorTaken);
  Edge edgeNotTaken = std::make_pair(root, successorNotTaken);

  // The new probability of those edges.
  double probTaken = branchHeuristicsInfo_->getProbabilityTaken(heuristic);
  double probNotTaken = branchHeuristicsInfo_->getProbabilityNotTaken(heuristic);

  // The old probability of those edges.
  double oldProbTaken    = getEdgeProbability(edgeTaken);
  double oldProbNotTaken = getEdgeProbability(edgeNotTaken);

  // Combined the newly matched heuristic with the already given
  // probability of an edge. Uses the Dempster-Shafer theory to combine
  // probability of two events to occur simultaneously.
  double d = oldProbTaken    * probTaken +
             oldProbNotTaken * probNotTaken;

  edgeProbabilities_[edgeTaken] = oldProbTaken * probTaken / d;
  edgeProbabilities_[edgeNotTaken] = oldProbNotTaken * probNotTaken / d;
}

char BranchPredictionPass::ID = 0;

static RegisterPass<BranchPredictionPass> X("branch-prediction",
                "Predict branch probabilities", false, true);

}  // namespace cost
}  // namespace llvm
}  // namespace compy
