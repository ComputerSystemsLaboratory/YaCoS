/*
 This file is distributed under the University of Illinois Open Source
 License. See LICENSE for details.
*/

#include "branch_heuristics_info.h"

using namespace ::llvm;

namespace compy {
namespace llvm {
namespace wlcost {

// The list of all heuristics with their respective probabilities.
// Notice that the list respect the order given in the ProfileHeuristics
// enumeration. This order will be used to index this list.
const struct BranchProbabilities
  BranchHeuristicsInfo::probList[BranchHeuristicsInfo::numBranchHeuristics_] = {
  { LOOP_BRANCH_HEURISTIC, 0.88f, 0.12f, "Loop Branch Heuristic" },
  { POINTER_HEURISTIC,     0.60f, 0.40f, "Pointer Heuristic"     },
  { CALL_HEURISTIC,        0.78f, 0.22f, "Call Heuristic"        },
  { OPCODE_HEURISTIC,      0.84f, 0.16f, "Opcode Heuristic"      },
  { LOOP_EXIT_HEURISTIC,   0.80f, 0.20f, "Loop Exit Heuristic"   },
  { RETURN_HEURISTIC,      0.72f, 0.28f, "Return Heuristic"      },
  { STORE_HEURISTIC,       0.55f, 0.45f, "Store Heuristic"       },
  { LOOP_HEADER_HEURISTIC, 0.75f, 0.25f, "Loop Header Heuristic" },
  { GUARD_HEURISTIC,       0.62f, 0.38f, "Guard Heuristic"       },
};

BranchHeuristicsInfo::BranchHeuristicsInfo(BranchPredictionInfo *BPI) {
  this->branchPredictionInfo_ = BPI;
  this->postDominatorTree_ = branchPredictionInfo_->getPostDominatorTree();
  this->loopInfo_ = branchPredictionInfo_->getLoopInfo();

  empty = std::make_pair((BasicBlock *) NULL, (BasicBlock *) NULL);
}

/// MatchHeuristic - Wrapper for the heuristics handlers meet above.
/// This procedure assumes that root basic block has exactly two successors.
/// @returns a Prediction that is a pair in which the first element is the
/// successor taken, and the second the successor not taken.
Prediction BranchHeuristicsInfo::matchHeuristic(BranchHeuristics bh, BasicBlock *root) const {
  // Try to match the heuristic bh with their respective handler.
  switch (bh) {
    case LOOP_BRANCH_HEURISTIC:
      return matchLoopBranchHeuristic(root);
    case POINTER_HEURISTIC:
      return matchPointerHeuristic(root);
    case CALL_HEURISTIC:
      return matchCallHeuristic(root);
    case OPCODE_HEURISTIC:
      return matchOpcodeHeuristic(root);
    case LOOP_EXIT_HEURISTIC:
      return matchLoopExitHeuristic(root);
    case RETURN_HEURISTIC:
      return matchReturnHeuristic(root);
    case STORE_HEURISTIC:
      return matchStoreHeuristic(root);
    case LOOP_HEADER_HEURISTIC:
      return matchLoopHeaderHeuristic(root);
    case GUARD_HEURISTIC:
      return matchGuardHeuristic(root);
  }
}

/// MatchLoopBranchHeuristic - Predict as taken an edge back to a loop's
/// head. Predict as not taken an edge exiting a loop.
/// @returns a Prediction that is a pair in which the first element is the
/// successor taken, and the second the successor not taken.
Prediction BranchHeuristicsInfo::matchLoopBranchHeuristic(BasicBlock *root) const {
  bool matched = false;
  Prediction pred;

  // Last instruction of basic block.
  Instruction *TI = root->getTerminator();

  // Basic block successors. True and False branches.
  BasicBlock *trueSuccessor = TI->getSuccessor(0);
  BasicBlock *falseSuccessor = TI->getSuccessor(1);

  // True and false branch edges.
  Edge trueEdge = std::make_pair(root, trueSuccessor);
  Edge falseEdge = std::make_pair(root, falseSuccessor);

  // If the true branch is a back edge to a loop's head or the false branch is
  // an exit edge, match the heuristic.
  if ((branchPredictionInfo_->isBackEdge(trueEdge) && loopInfo_->isLoopHeader(trueSuccessor)) ||
      branchPredictionInfo_->isExitEdge(falseEdge)) {
    matched = true;
    pred = std::make_pair(trueSuccessor, falseSuccessor);
  }

  // Check the opposite situation, the other branch.
  if ((branchPredictionInfo_->isBackEdge(falseEdge) && loopInfo_->isLoopHeader(falseSuccessor)) ||
      branchPredictionInfo_->isExitEdge(trueEdge)) {
    // If the heuristic matches both branches, predict none.
    if (matched)
      return empty;

    matched = true;
    pred = std::make_pair(falseSuccessor, trueSuccessor);
  }

  return (matched ? pred : empty);
}

/// MatchPointerHeuristic - Predict that a comparison of a pointer against
/// null or of two pointers will fail.
/// @returns a Prediction that is a pair in which the first element is the
/// successor taken, and the second the successor not taken.
Prediction BranchHeuristicsInfo::matchPointerHeuristic(BasicBlock *root) const {
  // Last instruction of basic block.
  Instruction *TI = root->getTerminator();

  // Basic block successors. True and False branches.
  BasicBlock *trueSuccessor = TI->getSuccessor(0);
  BasicBlock *falseSuccessor = TI->getSuccessor(1);

  // Is the last instruction a Branch Instruction?
  BranchInst *BI = dyn_cast<BranchInst>(TI);
  if (!BI || !BI->isConditional())
    return empty;

  // Conditional instruction.
  Value *cond = BI->getCondition();

  // Pointer comparisons are integer comparisons.
  ICmpInst *II = dyn_cast<ICmpInst>(cond);
  if (!II)
    return empty;

  // An integer comparison has always to operands.
  Value *operand1 = II->getOperand(0);
  Value *operand2 = II->getOperand(1);

  // Obtain the type of comparison.
  enum ICmpInst::Predicate signedPred = II->getSignedPredicate();

  // The heuristic states that it must be compared against null,
  // but in LLVM, null is also a PointerType, so it only requires
  // to test if there is a comparison between two pointers.
  if (signedPred == ICmpInst::ICMP_EQ &&
      isa<PointerType>(operand1->getType()) && // NULL is a pointer type too
      isa<PointerType>(operand2->getType())) { // NULL is a pointer type too
    return std::make_pair(falseSuccessor, trueSuccessor);
  } else if (signedPred != ICmpInst::ICMP_EQ &&
             isa<PointerType>(operand1->getType()) &&
             isa<PointerType>(operand2->getType())) {
    return std::make_pair(trueSuccessor, falseSuccessor);
  }

  return empty;
}

/// MatchCallHeuristic - Predict a successor that contains a call and does
/// not post-dominate will not be taken.
/// @returns a Prediction that is a pair in which the first element is the
/// successor taken, and the second the successor not taken.
Prediction BranchHeuristicsInfo::matchCallHeuristic(BasicBlock *root) const {
  bool matched = false;
  Prediction pred;

  // Last instruction of basic block.
  Instruction *TI = root->getTerminator();

  // Basic block successors. True and False branches.
  BasicBlock *trueSuccessor = TI->getSuccessor(0);
  BasicBlock *falseSuccessor = TI->getSuccessor(1);

  // Check if the successor contains a call and does not post-dominate.
  if (branchPredictionInfo_->hasCall(trueSuccessor) &&
      !postDominatorTree_->dominates(trueSuccessor, root)) {
    matched = true;
    pred = std::make_pair(falseSuccessor, trueSuccessor);
  }

  // Check the opposite situation, the other branch.
  if (branchPredictionInfo_->hasCall(falseSuccessor) &&
      !postDominatorTree_->dominates(falseSuccessor, root)) {
    // If the heuristic matches both branches, predict none.
    if (matched)
      return empty;

    matched = true;
    pred = std::make_pair(trueSuccessor, falseSuccessor);
  }

  return (matched ? pred : empty);
}

/// MatchOpcodeHeuristic - Predict that a comparison of an integer for less
/// than zero, less than or equal to zero, or equal to a constant, will
/// fail.
/// @returns a Prediction that is a pair in which the first element is the
/// successor taken, and the second the successor not taken.
Prediction BranchHeuristicsInfo::matchOpcodeHeuristic(BasicBlock *root) const {
  // Last instruction of basic block.
  Instruction *TI = root->getTerminator();

  // Basic block successors, the true and false branches.
  BasicBlock *trueSuccessor = TI->getSuccessor(0);
  BasicBlock *falseSuccessor = TI->getSuccessor(1);

  // Is the last instruction a Branch Instruction?
  BranchInst *BI = dyn_cast<BranchInst>(TI);
  if (!BI || !BI->isConditional())
    return empty;

  // Conditional instruction.
  Value *cond = BI->getCondition();

  // Heuristics can only apply to integer comparisons.
  ICmpInst *II = dyn_cast<ICmpInst>(cond);
  if (!II)
    return empty;

  // An integer comparison has always to operands.
  Value *operand1 = II->getOperand(0);
  ConstantInt *op1const = dyn_cast<ConstantInt>(operand1);

  Value *operand2 = II->getOperand(1);
  ConstantInt *op2const = dyn_cast<ConstantInt>(operand2);

  // The type of comparison used.
  enum ICmpInst::Predicate pred = II->getUnsignedPredicate();

  // The return successors (the first taken and the second not taken).
  Edge falseEdge = std::make_pair(falseSuccessor, trueSuccessor);
  Edge trueEdge  = std::make_pair(trueSuccessor, falseSuccessor);

  // Check several comparison operators.
  switch (pred) {
    case ICmpInst::ICMP_EQ: // if ($var == constant) or if (constant == $var).
      // If it's a equal comparison against a constant integer, match.
      if (op1const || op2const)
        return falseEdge;

      break;
    case ICmpInst::ICMP_NE: // if ($var != constant) or if (constant != $var).
      // If it's a not equal comparison against a constant integer, match.
      if (op1const || op2const)
        return trueEdge;

      break;
    case ICmpInst::ICMP_SLT: // if ($var < 0) or if (0 < $var).
    case ICmpInst::ICMP_ULT:
      if (!op1const && (op2const && op2const->isZero()))
        return falseEdge;
      else if (!op2const && (op1const && op1const->isZero()))
        return trueEdge;

      break;
    case ICmpInst::ICMP_SLE: // if ($var <= 0) or if (0 <= $var).
    case ICmpInst::ICMP_ULE:
      if (!op1const && (op2const && op2const->isZero()))
        return falseEdge;
      else if (!op2const && (op1const && op1const->isZero()))
        return trueEdge;

      break;
    case ICmpInst::ICMP_SGT: // if ($var > 0) or if (0 > $var).
    case ICmpInst::ICMP_UGT:
      if (!op1const && (op2const && op2const->isZero()))
        return trueEdge;
      else if (!op2const && (op1const && op1const->isZero()))
        return falseEdge;

      break;
    case ICmpInst::ICMP_SGE: // if ($var >= 0) or if (0 >= $var).
    case ICmpInst::ICMP_UGE:
      if (!op1const && (op2const && op2const->isZero()))
        return trueEdge;
      else if (!op2const && (op1const && op1const->isZero()))
        return falseEdge;

      break;
    default: // Do not process any other comparison operators.
      break;
  }

  // Heuristic not matched.
  return empty;
}

/// MatchLoopExitHeuristic - Predict that a comparison in a loop in which no
/// successor is a loop head will not exit the loop.
/// @returns a Prediction that is a pair in which the first element is the
/// successor taken, and the second the successor not taken.
Prediction BranchHeuristicsInfo::matchLoopExitHeuristic(BasicBlock *root) const {
  // Last instruction of basic block.
  Instruction *TI = root->getTerminator();

  // Basic block successors. True and False branches.
  BasicBlock *trueSuccessor = TI->getSuccessor(0);
  BasicBlock *falseSuccessor = TI->getSuccessor(1);

  // Get the most inner loop in which this basic block is in.
  Loop *loop = loopInfo_->getLoopFor(root);

  // If there's a loop, check if neither of the branches are loop headers.
  if (!loop || loopInfo_->isLoopHeader(trueSuccessor) ||
      loopInfo_->isLoopHeader(falseSuccessor))
    return empty;

  // True and false branch edges.
  Edge trueEdge = std::make_pair(root, trueSuccessor);
  Edge falseEdge = std::make_pair(root, falseSuccessor);

  // If it is an exit edge, successor will fail so predict the other branch.
  // Note that is not possible for both successors to be exit edges.
  if (branchPredictionInfo_->isExitEdge(trueEdge))
    return std::make_pair(falseSuccessor, trueSuccessor);
  else if (branchPredictionInfo_->isExitEdge(falseEdge))
    return std::make_pair(trueSuccessor, falseSuccessor);

  return empty;
}

/// MatchReturnHeuristic - Predict a successor that contains a return will
/// not be taken.
/// @returns a Prediction that is a pair in which the first element is the
/// successor taken, and the second the successor not taken.
Prediction BranchHeuristicsInfo::matchReturnHeuristic(BasicBlock *root) const {
  bool matched = false;
  Prediction pred;

  // Last instruction of basic block.
  Instruction *TI = root->getTerminator();

  // Basic block successors. True and False branches.
  BasicBlock *trueSuccessor = TI->getSuccessor(0);
  BasicBlock *falseSuccessor = TI->getSuccessor(1);

  // Check if the true successor it's a return instruction.
  if (isa<ReturnInst>(trueSuccessor->getTerminator())) {
    matched = true;
    pred = std::make_pair(falseSuccessor, trueSuccessor);
  }

  // Check the opposite situation, the other branch.
  if (isa<ReturnInst>(falseSuccessor->getTerminator())) {
    // If the heuristic matches both branches, predict none.
    if (matched)
      return empty;

    matched = true;
    pred = std::make_pair(trueSuccessor, falseSuccessor);
  }

  return (matched ? pred : empty);
}

/// MatchStoreHeuristic - Predict a successor that contains a store
/// instruction and does not post-dominate will not be taken.
/// @returns a Prediction that is a pair in which the first element is the
/// successor taken, and the second the successor not taken.
Prediction BranchHeuristicsInfo::matchStoreHeuristic(BasicBlock *root) const {
  bool matched = false;
  Prediction pred;

  // Last instruction of basic block.
  Instruction *TI = root->getTerminator();

  // Basic block successors. True and False branches.
  BasicBlock *trueSuccessor = TI->getSuccessor(0);
  BasicBlock *falseSuccessor = TI->getSuccessor(1);

  // Check if the successor contains a store and does not post-dominate.
  if (branchPredictionInfo_->hasStore(trueSuccessor) &&
      !postDominatorTree_->dominates(trueSuccessor, root)) {
    matched = true;
    pred = std::make_pair(falseSuccessor, trueSuccessor);
  }

  // Check the opposite situation, the other branch.
  if (branchPredictionInfo_->hasStore(falseSuccessor) &&
      !postDominatorTree_->dominates(falseSuccessor, root)) {
    // If the heuristic matches both branches, predict none.
    if (matched)
      return empty;

    matched = true;
    pred = std::make_pair(trueSuccessor, falseSuccessor);
  }

  return (matched ? pred : empty);
}

/// MatchLoopHeaderHeuristic - Predict a successor that is a loop header or
/// a loop pre-header and does not post-dominate will be taken.
/// @returns a Prediction that is a pair in which the first element is the
/// successor taken, and the second the successor not taken.
Prediction BranchHeuristicsInfo::matchLoopHeaderHeuristic(BasicBlock *root) const {
  bool matched = false;
  Prediction pred;

  // Last instruction of basic block.
  Instruction *TI = root->getTerminator();

  // Basic block successors. True and False branches.
  BasicBlock *trueSuccessor = TI->getSuccessor(0);
  BasicBlock *falseSuccessor = TI->getSuccessor(1);

  // Get the most inner loop in which the true successor basic block is in.
  Loop *loop = loopInfo_->getLoopFor(trueSuccessor);

  // Check if exists a loop, the true branch successor is a loop header or a
  // loop pre-header, and does not post dominate.
  if (loop && (trueSuccessor == loop->getHeader() ||
      trueSuccessor == loop->getLoopPreheader()) &&
      !postDominatorTree_->dominates(trueSuccessor, root)) {
    matched = true;
    pred = std::make_pair(trueSuccessor, falseSuccessor);
  }

  // Get the most inner loop in which the false successor basic block is in.
  loop = loopInfo_->getLoopFor(falseSuccessor);

  // Check if exists a loop,
  // the false branch successor is a loop header or a loop pre-header, and
  // does not post dominate.
  if (loop && (falseSuccessor == loop->getHeader() ||
      falseSuccessor == loop->getLoopPreheader()) &&
      !postDominatorTree_->dominates(falseSuccessor, root)) {
    // If the heuristic matches both branches, predict none.
    if (matched)
      return empty;

    matched = true;
    pred = std::make_pair(falseSuccessor, trueSuccessor);
  }

  return (matched ? pred : empty);
}

/// MatchGuardHeuristic - Predict that a comparison in which a register is
/// an operand, the register is used before being defined in a successor
/// block, and the successor block does not post-dominate will reach the
/// successor block.
/// @returns a Prediction that is a pair in which the first element is the
/// successor taken, and the second the successor not taken.
Prediction BranchHeuristicsInfo::matchGuardHeuristic(BasicBlock *root) const {
  bool matched = false;
  Prediction pred;

  // Last instruction of basic block.
  Instruction *TI = root->getTerminator();

  // Basic block successors. True and False branches.
  BasicBlock *trueSuccessor = TI->getSuccessor(0);
  BasicBlock *falseSuccessor = TI->getSuccessor(1);

  // Is the last instruction a Branch Instruction?
  BranchInst *BI = dyn_cast<BranchInst>(TI);
  if (!BI || !BI->isConditional())
    return empty;

  // Conditional instruction.
  Value *cond = BI->getCondition();

  // Find if the variable used in the branch instruction is
  // in fact a comparison instruction.
  CmpInst *CI = dyn_cast<CmpInst>(cond);
  if (!CI)
    return empty;

  // Seek over all of the operands of this comparison instruction.
  for (unsigned ops = 0; ops < CI->getNumOperands(); ++ops) {
    // Find the operand.
    Value *operand = CI->getOperand(ops);

    // Check if the operand is a function argument or a value.
    if (!isa<Argument>(operand) && !isa<User>(operand))
      continue;

    // Check if this variable was used in the true successor and
    // does not post dominate.
    // Since LLVM is in SSA form, it's impossible for a variable being used
    // before being defined, so that statement is skipped.
    if (operand->isUsedInBasicBlock(trueSuccessor) &&
        !postDominatorTree_->dominates(trueSuccessor, root)) {
      // If a heuristic was already matched, predict none and abort immediately.
      if (matched)
        return empty;

      matched = true;
      pred = std::make_pair(trueSuccessor, falseSuccessor);
    }

    // Check if this variable was used in the false successor and
    // does not post dominate.
    if (operand->isUsedInBasicBlock(falseSuccessor) &&
        !postDominatorTree_->dominates(falseSuccessor, root)) {
      // If a heuristic was already matched, predict none and abort immediately.
      if (matched)
        return empty;

      matched = true;
      pred = std::make_pair(falseSuccessor, trueSuccessor);
    }
  }

  return (matched ? pred : empty);
}

}  // namespace cost
}  // namespace llvm
}  // namespace compy
